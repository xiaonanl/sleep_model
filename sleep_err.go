// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// hip runs a hippocampus model on the Close-Far paired associate learning task
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/split"
	"github.com/emer/leabra/hip"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// this is the stub main for gogi that calls our actual mainrun function, at end of file
func main() {
	TheSim.New()
	TheSim.Config()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "keeping default params for generic prjns",
				Params: params.Params{
					"Prjn.Learn.Momentum.On": "true",
					"Prjn.Learn.Norm.On":     "true",
					"Prjn.Learn.WtBal.On":    "false",
				}},
			{Sel: ".EcCa1Prjn", Desc: "encoder projections -- no norm, moment",
				Params: params.Params{
					"Prjn.Learn.Lrate":        "0.04",
					"Prjn.Learn.Momentum.On":  "false",
					"Prjn.Learn.Norm.On":      "false",
					"Prjn.Learn.WtBal.On":     "true",
					"Prjn.Learn.XCal.SetLLrn": "false", // using bcm now, better
				}},
			{Sel: ".HippoCHL", Desc: "hippo CHL projections -- no norm, moment, but YES wtbal = sig better",
				Params: params.Params{
					"Prjn.CHL.Hebb":          "0.05",
					"Prjn.Learn.Lrate":       "0.2",
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.WtBal.On":    "true",
				}},
			{Sel: ".PPath", Desc: "perforant path, new Dg error-driven EcCa1Prjn prjns",
				Params: params.Params{
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.WtBal.On":    "true",
					"Prjn.Learn.Lrate":       "0.15", // err driven: .15 > .2 > .25 > .1
					// moss=4, delta=4, lr=0.2, test = 3 are best
				}},
			{Sel: "#CA1ToECout", Desc: "extra strong from CA1 to ECout",
				Params: params.Params{
					"Prjn.WtScale.Abs": "4.0",
				}},
			{Sel: "#InputToECin", Desc: "one-to-one input to EC",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.8",
					"Prjn.WtInit.Var":  "0.0",
				}},
			{Sel: "#ECoutToECin", Desc: "one-to-one out to in",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.9",
					"Prjn.WtInit.Var":  "0.01",
					"Prjn.WtScale.Rel": "0.5",
				}},
			{Sel: "#DGToCA3", Desc: "Mossy fibers: strong, non-learning",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.9",
					"Prjn.WtInit.Var":  "0.01",
					"Prjn.WtScale.Rel": "4",
				}},
			{Sel: "#CA3ToCA3", Desc: "CA3 recurrent cons",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.1",
					"Prjn.Learn.Lrate": "0.1",
				}},
			{Sel: "#ECinToDG", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
				Params: params.Params{
					"Prjn.Learn.Learn":       "true", // absolutely essential to have on!
					"Prjn.CHL.Hebb":          ".5",   // .5 > 1 overall
					"Prjn.CHL.SAvgCor":       "0.1",  // .1 > .2 > .3 > .4 ?
					"Prjn.CHL.MinusQ1":       "true", // dg self err?
					"Prjn.Learn.Lrate":       "0.4",  // .4 > .3 > .2
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.WtBal.On":    "true",
				}},
			{Sel: "#CA3ToCA1", Desc: "Schaffer collaterals -- slower, less hebb",
				Params: params.Params{
					"Prjn.CHL.Hebb":          "0.01",
					"Prjn.CHL.SAvgCor":       "0.4",
					"Prjn.Learn.Lrate":       "0.1",
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.WtBal.On":    "true",
				}},

			{Sel: "#ECin", Desc: "EC in -- with layer-level",
				Params: params.Params{
					"Layer.Act.Gbar.L":         ".1",
					"Layer.Inhib.ActAvg.Init":  "0.16",
					"Layer.Inhib.Layer.On":     "true",
					"Layer.Inhib.Layer.Gi":     "0.8",
					"Layer.Inhib.Pool.Gi":      "2.0",
					"Layer.Inhib.Pool.On":      "true",
					"Layer.Inhib.ActAvg.Fixed": "false",
				}},
			{Sel: "#ECoutToOutput", Desc: "ECintoOUTPUT",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.5",
					"Prjn.Learn.Learn": "false",
				}},

			{Sel: "#InputToOutput", Desc: "InputtoOUTPUT",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.5",
					"Prjn.Learn.Learn": "false",
				}},

			{Sel: "#OutputToECin", Desc: "cortical interference",
				Params: params.Params{
					"Prjn.WtScale.Rel": "1",
					"Prjn.Learn.Learn": "false",
				}},
			{Sel: "#Output", Desc: "Output final",
				Params: params.Params{
					"Layer.Act.Gbar.L":         ".1",
					"Layer.Inhib.ActAvg.Init":  "0.16",
					"Layer.Inhib.Layer.On":     "true",
					"Layer.Inhib.Layer.Gi":     "1.5",
					"Layer.Inhib.Pool.Gi":      "2.0",
					"Layer.Inhib.Pool.On":      "true",
					"Layer.Inhib.ActAvg.Fixed": "false",
				}},
			{Sel: "#Cortex", Desc: "using default 1.8 inhib for all of network -- can explore",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":     "1.8",
					"Layer.Inhib.ActAvg.Init":  "0.088",
					"Layer.Inhib.ActAvg.Fixed": "false",
				}},
			{Sel: "#Semantic", Desc: "using default 1.8 inhib for all of network -- can explore",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
				}},

			{Sel: "#Letter", Desc: "using default 1.8 inhib for all of network -- can explore",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":     "1.8",
					"Layer.Inhib.ActAvg.Init":  "0.095",
					"Layer.Inhib.ActAvg.Fixed": "false",
				}},

			{Sel: "#Input", Desc: "using default 1.8 inhib for all of network -- can explore",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":     "1.8",
					"Layer.Inhib.ActAvg.Init":  "0.15",
					"Layer.Inhib.ActAvg.Fixed": "false",
				}},

			{Sel: "#ECout", Desc: "EC out: only pools, no layer-level",
				Params: params.Params{
					"Layer.Act.Gbar.L":         ".1",
					"Layer.Inhib.Layer.On":     "false",
					"Layer.Inhib.Layer.Gi":     "0.8",
					"Layer.Inhib.Pool.Gi":      "2.0",
					"Layer.Inhib.Pool.On":      "true",
					"Layer.Inhib.ActAvg.Init":  "0.16",
					"Layer.Inhib.ActAvg.Fixed": "false",
				}},
			{Sel: "#DG", Desc: "very sparse = high inibhition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.01",
					"Layer.Inhib.Layer.Gi":    "3.8",
				}},
			{Sel: "#CA3", Desc: "sparse = high inibhition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.02",
					"Layer.Inhib.Layer.Gi":    "2.8",
				}},
			{Sel: "#CA1", Desc: "CA1 only Pools",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.1",
					"Layer.Inhib.Layer.On":    "false",
					"Layer.Inhib.Pool.Gi":     "2.4",
					"Layer.Inhib.Pool.On":     "true",
				}},
		},
	}},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *leabra.Network `view:"no-inline"`
	TrainClose   *etable.Table   `view:"no-inline" desc:"Close training patterns to use"`
	TrainFar     *etable.Table   `view:"no-inline" desc:"Far training patterns to use"`
	TestClose    *etable.Table   `view:"no-inline" desc:"Close testing patterns to use"`
	TestFar      *etable.Table   `view:"no-inline" desc:"Far testing patterns to use"`
	TrnTrlLog    *etable.Table   `view:"no-inline" desc:"training trial-level log data"`
	TrnEpcLog    *etable.Table   `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog    *etable.Table   `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog    *etable.Table   `view:"no-inline" desc:"testing trial-level log data"`
	TstErrLog    *etable.Table   `view:"no-inline" desc:"log of all test trials where errors were made"`
	TstErrStats  *etable.Table   `view:"no-inline" desc:"stats on test trials where errors were made"`
	TstCycLog    *etable.Table   `view:"no-inline" desc:"testing cycle-level log data"`
	RunLog       *etable.Table   `view:"no-inline" desc:"summary log of each run"`
	RunStats     *etable.Table   `view:"no-inline" desc:"aggregate stats on all runs"`
	TstStats     *etable.Table   `view:"no-inline" desc:"testing stats"`
	Params       params.Sets     `view:"no-inline" desc:"full collection of param sets"`
	ParamSet     string          `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	Tag          string          `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params)"`
	MaxRuns      int             `desc:"maximum number of model runs to perform"`
	MaxEpcs      int             `desc:"maximum number of epochs to run per model run"`
	MaxSubs      int             `desc:"maximum number of epochs to run per model run"`
	Group        int
	SemWake      int
	Outecback    float64
	TrainEnv     env.FixedTable    `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestEnv      env.FixedTable    `desc:"Testing environment -- manages iterating over testing"`
	Time         leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn       bool              `desc:"whether to update the network view while running"`
	TrainUpdt    leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt     leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval int               `desc:"how often to run through all the test patterns, in terms of training epochs"`
	MemThr       float64           `desc:"threshold to use for memory test -- if error proportion is below this number, it is scored as a correct trial"`

	// statistics: note use float64 as that is best for etable.Table
	TestNm         string  `inactive:"+" desc:"what set of patterns are we currently testing"`
	Mem            float64 `inactive:"+" desc:"whether current trial's ECout met memory criterion"`
	TrgOnWasOffAll float64 `inactive:"+" desc:"current trial's proportion of bits where target = on but ECout was off ( < 0.5), for all bits"`
	TrgOnWasOffCmp float64 `inactive:"+" desc:"current trial's proportion of bits where target = on but ECout was off ( < 0.5), for only completion bits that were not active in ECin"`
	TrgOffWasOn    float64 `inactive:"+" desc:"current trial's proportion of bits where target = off but ECout was on ( > 0.5)"`
	TrgOnWasOffB   float64 `inactive:"+" desc:"current trial's proportion of bits in B pool where target = on but ECout was off ( < 0.5), for only completion bits that were not active in ECin"`
	TrgOnWasOffC   float64 `inactive:"+" desc:"current trial's proportion of bits in C pool where target = on but ECout was off ( < 0.5), for only completion bits that were not active in ECin"`
	TrgOffWasOnB   float64 `inactive:"+" desc:"current trial's proportion of bits in B pool where target = off but ECout was on ( > 0.5), for only completion bits that were not active in ECin"`
	TrgOffWasOnC   float64 `inactive:"+" desc:"current trial's proportion of bits in C pool where target = off but ECout was on ( > 0.5), for only completion bits that were not active in ECin"`
	TrlSSEB        float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlSSEC        float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSEB     float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlAvgSSEC     float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff     float64 `inactive:"+" desc:"current trial's cosine difference"`

	EpcSSE     float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcAvgSSE  float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr  float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)"`
	EpcPctCor  float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"`
	EpcCosDiff float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	FirstZero  int     `inactive:"+" desc:"epoch at when SSE first went to zero"`

	// internal state - view:"-"
	SumSSE       float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE    float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff   float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	CntErr       int              `view:"-" inactive:"+" desc:"sum of errs to increment as we go through epoch"`
	Win          *gi.Window       `view:"-" desc:"main GUI window"`
	NetView      *netview.NetView `view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar      `view:"-" desc:"the master toolbar"`
	TrnTrlPlot   *eplot.Plot2D    `view:"-" desc:"the training trial plot"`
	TrnEpcPlot   *eplot.Plot2D    `view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D    `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D    `view:"-" desc:"the test-trial plot"`
	TstCycPlot   *eplot.Plot2D    `view:"-" desc:"the test-cycle plot"`
	RunPlot      *eplot.Plot2D    `view:"-" desc:"the run plot"`
	TrnEpcFile   *os.File         `view:"-" desc:"log file"`
	RunFile      *os.File         `view:"-" desc:"log file"`
	TstTrialFile *os.File         `view:"-" desc:"log file"`
	TmpVals      []float32        `view:"-" desc:"temp slice for holding values -- prevent mem allocs"`
	LayStatNms   []string         `view:"-" desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	TstNms       []string         `view:"-" desc:"names of test tables"`
	TstStatNms   []string         `view:"-" desc:"names of test stats"`
	Daydream     bool
	SaveWts      bool  `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui        bool  `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool  `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool  `view:"-" desc:"true if sim is running"`
	StopNow      bool  `view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool  `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed      int64 `view:"-" desc:"the current random seed"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.TrainClose = &etable.Table{}
	ss.TrainFar = &etable.Table{}
	ss.TestClose = &etable.Table{}
	ss.TestFar = &etable.Table{}
	ss.TrnTrlLog = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.TstCycLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = ParamSets
	//ss.Params = SavedParamsSets
	ss.RndSeed = 1
	ss.ViewOn = true
	ss.TrainUpdt = leabra.AlphaCycle
	ss.TestUpdt = leabra.Cycle
	ss.TestInterval = 1000
	ss.LogSetParams = false
	ss.MemThr = 0.34
	ss.LayStatNms = []string{"ECin", "DG", "CA3", "CA1"}
	ss.TstNms = []string{"Close", "Far"}
	ss.TstStatNms = []string{"Mem", "TrgOnWasOff", "TrgOffWasOn"}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstCycLog(ss.TstCycLog)
	ss.ConfigRunLog(ss.RunLog)
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 1
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 50
	}
	if ss.MaxSubs == 0 { // allow user override
		ss.MaxSubs = 5
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainClose)
	ss.TrainEnv.Sequential = true
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestNm = "Close"
	ss.TestEnv.Table = etable.NewIdxView(ss.TestClose)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

// SetEnv select which set of patterns to train on: Close or Far
func (ss *Sim) SetEnv(TrainFar bool) {
	if TrainFar {
		ss.TrainEnv.Table = etable.NewIdxView(ss.TrainFar)
		ss.TestNm = "Far"
		ss.TestEnv.Table = etable.NewIdxView(ss.TestFar)
	} else {
		ss.TrainEnv.Table = etable.NewIdxView(ss.TrainClose)
		ss.TestNm = "Close"
		ss.TestEnv.Table = etable.NewIdxView(ss.TestClose)
	}
	ss.TrainEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "Hip")
	in := net.AddLayer4D("Input", 3, 2, 7, 7, emer.Input)
	cortex := net.AddLayer2D("Cortex", 20, 20, emer.Hidden)
	sem := net.AddLayer2D("Semantic", 14, 14, emer.Hidden)
	letterhid := net.AddLayer2D("Letter", 14, 14, emer.Hidden)
	ecin := net.AddLayer4D("ECin", 3, 2, 7, 7, emer.Hidden)
	ecout := net.AddLayer4D("ECout", 3, 2, 7, 7, emer.Target) // clamped in plus phase
	ca1 := net.AddLayer4D("CA1", 3, 2, 30, 35, emer.Hidden)
	dg := net.AddLayer2D("DG", 70, 70, emer.Hidden)
	ca3 := net.AddLayer2D("CA3", 50, 50, emer.Hidden)
	out := net.AddLayer4D("Output", 3, 2, 7, 7, emer.Target)

	ecin.SetClass("EC")
	ecout.SetClass("EC")

	sem.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Input", YAlign: relpos.Front, Space: 2})
	cortex.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Semantic", YAlign: relpos.Front, Space: 2})
	letterhid.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Cortex", YAlign: relpos.Front, Space: 2})
	out.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Letter", YAlign: relpos.Front, Space: 2})
	ecin.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Input", YAlign: relpos.Front, XAlign: relpos.Right, Space: 0})
	ecout.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "ECin", YAlign: relpos.Front, Space: 2})
	dg.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "ECin", YAlign: relpos.Front, XAlign: relpos.Left, Space: 2})
	ca3.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "DG", YAlign: relpos.Front, XAlign: relpos.Left, Space: 0})
	ca1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "CA3", YAlign: relpos.Front, Space: 2})

	onetoone := prjn.NewOneToOne()
	pool1to1 := prjn.NewPoolOneToOne()
	full := prjn.NewFull()

	net.ConnectLayers(in, ecin, onetoone, emer.Forward)
	net.ConnectLayers(ecout, ecin, onetoone, emer.Back)
	net.ConnectLayers(ecout, out, onetoone, emer.Forward)
	net.ConnectLayers(out, ecin, onetoone, emer.Forward)
	//	net.ConnectLayers(in, out, prjn.NewOneToOne(), emer.Forward)
	//Input to Cortex
	IntoECin := prjn.NewPoolOneToOne()
	IntoECin.SendStart = 0
	IntoECin.SendStart2 = 4
	IntoECin.NPools = 1
	IntoECin.NPools2 = 2

	IntoLetter := prjn.NewPoolOneToOne()
	IntoLetter.SendStart = 1
	IntoLetter.SendStart2 = 1
	IntoLetter.NPools = 1
	IntoLetter.NPools2 = 0

	IntoSem := prjn.NewPoolOneToOne()
	IntoSem.SendStart = 2
	IntoSem.SendStart2 = 2
	IntoSem.NPools = 2
	IntoSem.NPools2 = 0

	net.ConnectLayers(in, cortex, IntoECin, emer.Forward)
	net.ConnectLayers(in, letterhid, IntoLetter, emer.Forward)
	net.ConnectLayers(in, sem, IntoSem, emer.Forward)

	CortextoECin := prjn.NewPoolOneToOne()
	CortextoECin.SendStart = 2
	CortextoECin.SendStart2 = 2
	CortextoECin.RecvStart = 2
	CortextoECin.NPools = 2
	CortextoECin.NPools2 = 0
	net.BidirConnectLayers(cortex, out, CortextoECin)
	//net.BidirConnectLayers(semantic, ecin, CortextoECin)
	net.BidirConnectLayers(letterhid, out, CortextoECin)
	net.BidirConnectLayers(sem, ecin, CortextoECin)

	// EC <-> CA1 encoder pathways
	pj := net.ConnectLayersPrjn(ecin, ca1, pool1to1, emer.Forward, &hip.EcCa1Prjn{})
	pj.SetClass("EcCa1Prjn")
	pj = net.ConnectLayersPrjn(ca1, ecout, pool1to1, emer.Forward, &hip.EcCa1Prjn{})
	pj.SetClass("EcCa1Prjn")
	pj = net.ConnectLayersPrjn(ecout, ca1, pool1to1, emer.Back, &hip.EcCa1Prjn{})
	pj.SetClass("EcCa1Prjn")

	// Perforant pathway
	ppath := prjn.NewUnifRnd()
	ppath.PCon = 0.25

	pj = net.ConnectLayersPrjn(ecin, dg, ppath, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("HippoCHL")

	pj = net.ConnectLayersPrjn(ecin, ca3, ppath, emer.Forward, &hip.EcCa1Prjn{})
	pj.SetClass("PPath")
	pj = net.ConnectLayersPrjn(ca3, ca3, full, emer.Lateral, &hip.EcCa1Prjn{})
	pj.SetClass("PPath")

	// Mossy fibers
	mossy := prjn.NewUnifRnd()
	mossy.PCon = 0.02
	pj = net.ConnectLayersPrjn(dg, ca3, mossy, emer.Forward, &hip.CHLPrjn{}) // no learning
	pj.SetClass("HippoCHL")

	// Schafer collaterals
	pj = net.ConnectLayersPrjn(ca3, ca1, full, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("HippoCHL")
	//schafer :=prjn.NewUnifRnd()
	//schafer.PCon = 0.50
	//pj = net.ConnectLayersPrjn(ca3, ca3, schafer, emer.Lateral, &hip.CHLPrjn{})
	//pj.SetClass("HippoCHL")
	//pj = net.ConnectLayersPrjn(ca3, ca1, schafer, emer.Forward, &hip.CHLPrjn{})
	//pj.SetClass("HippoCHL")

	// using 3 threads :)
	dg.SetThread(1)
	ca3.SetThread(2)
	ca1.SetThread(3)
	//ecin.SetThread(4)
	//ecout.SetThread(5)
	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// outLay.SetType(emer.Compare)
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

	net.Defaults()
	ss.SetParams("Network", ss.LogSetParams) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.StopNow = false
	ss.SetParams("", ss.LogSetParams) // all sheets
	ss.NewRun()
	ss.UpdateView(true)
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func (ss *Sim) NewRndSeed() {
	ss.RndSeed = time.Now().UnixNano()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters(train bool) string {
	if train {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName.Cur)
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName.Cur)
	}
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(train))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of AlphaCycle

func (ss *Sim) AllOn() {
	lays := []string{"Input", "ECout", "CA1", "CA3", "DG", "ECin", "Letter", "Cortex", "Semantic"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		ly.Off = false
	}
}

func (ss *Sim) AlphaCycHip(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.

	ca1 := ss.Net.LayerByName("CA1").(leabra.LeabraLayer).AsLeabra()
	ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	dg := ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()

	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	ecout := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	cortex := ss.Net.LayerByName("Cortex").(leabra.LeabraLayer).AsLeabra()
	letter := ss.Net.LayerByName("Letter").(leabra.LeabraLayer).AsLeabra()
	ca1FmECin := ca1.RcvPrjns.SendName("ECin").(*hip.EcCa1Prjn)
	ca1FmCa3 := ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn)
	ECinFMInput := ecin.RcvPrjns.SendName("Input").(*leabra.Prjn)
	ECinFMECout := ecin.RcvPrjns.SendName("ECout").(*leabra.Prjn)
	OutputFMCortex := out.RcvPrjns.SendName("Cortex").(*leabra.Prjn)
	OutputFMLetter := out.RcvPrjns.SendName("Letter").(*leabra.Prjn)
	OutputFMECin := out.RcvPrjns.SendName("ECout").(*leabra.Prjn)
	CortexFMOutput := cortex.RcvPrjns.SendName("Output").(*leabra.Prjn)
	ECinFMOutput := ecin.RcvPrjns.SendName("Output").(*leabra.Prjn)
	LetterFMOutput := letter.RcvPrjns.SendName("Output").(*leabra.Prjn)
	CortexFMin := cortex.RcvPrjns.SendName("Input").(*leabra.Prjn)
	ca3FmECin := ca3.RcvPrjns.SendName("ECin").(*hip.EcCa1Prjn)
	dgFmECin := dg.RcvPrjns.SendName("ECin").(*hip.CHLPrjn)
	letterhid := ss.Net.LayerByName("Letter").(leabra.LeabraLayer).AsLeabra()
	ca3Fmca3 := ca3.RcvPrjns.SendName("CA3").(*hip.EcCa1Prjn)
	ca1Fmca3 := ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn)
	ECoutFMca1 := ecout.RcvPrjns.SendName("CA1").(*hip.EcCa1Prjn)
	ca1FMECout := ca1.RcvPrjns.SendName("ECout").(*hip.EcCa1Prjn)
	ca3FmDg := ca3.RcvPrjns.SendName("DG").(leabra.LeabraPrjn).AsLeabra()

	// semantic learn switch
	sem := ss.Net.LayerByName("Semantic").(leabra.LeabraLayer).AsLeabra()
	ECinFMSemantic := ecin.RcvPrjns.SendName("Semantic").(*leabra.Prjn)
	SemanticFMInput := sem.RcvPrjns.SendName("Input").(*leabra.Prjn)
	SemanticFMECin := sem.RcvPrjns.SendName("ECin").(*leabra.Prjn)
	ECinFMSemantic.Learn.Learn = false
	SemanticFMInput.Learn.Learn = false
	SemanticFMECin.Learn.Learn = false
	ECinFMSemantic.WtScale.Rel = float32(ss.SemWake) / 100
	//ecin.Inhib.Layer.On = false
	ECinFMECout.WtScale.Rel = 0.5
	ECinFMOutput.WtScale.Abs = 0
	ECinFMOutput.WtScale.Rel = float32(ss.Outecback)
	//ca3Fmca3.CHL.On = true
	//ca1Fmca3.CHL.On = true
	//dgFmECin.CHL.On = true
	//ca3FmECin.CHL.On = true
	//ECinFMSemantic.Learn.Learn = false
	LetterFMOutput.Learn.Learn = false
	CortexFMOutput.Learn.Learn = false
	OutputFMCortex.Learn.Learn = false
	OutputFMLetter.Learn.Learn = false
	ca1FMECout.Learn.Learn = true
	ca3Fmca3.Learn.Learn = true
	ca1Fmca3.Learn.Learn = true
	ECoutFMca1.Learn.Learn = true
	ca3FmECin.Learn.Learn = true
	dgFmECin.Learn.Learn = true
	ca1FmECin.Learn.Learn = true
	cortex.Off = true
	OutputFMECin.Off = false
	letterhid.Off = true
	ECinFMInput.Off = false
	ECinFMECout.Off = false
	OutputFMLetter.Off = true
	OutputFMCortex.Off = true
	CortexFMOutput.Off = true
	CortexFMin.Off = true
	ca1FmECin.Off = false
	ca3FmECin.Off = false
	dgFmECin.Off = false
	ca1.Off = false
	ca3.Off = false
	dg.Off = false
	ecout.Off = false
	//dgFmECin.WtScale.Abs = 1
	//ca3FmECin.WtScale.Abs = 1

	//ca1.SetType(emer.Hidden)
	//ca3.SetType(emer.Hidden)
	//dg.SetType(emer.Hidden)
	dg.SetType(emer.Hidden)
	ca1.SetType(emer.Hidden)
	ca3.SetType(emer.Hidden)

	if train {
		ecin.SetType(emer.Hidden)
		ecout.SetType(emer.Target) // clamp a plus phase during testing
	} else {
		// don't clamp
		ecout.SetType(emer.Compare)
		//ecin.UpdateExtFlags() // call this after updating type
	}
	ecout.UpdateExtFlags() // call this after updating type

	// First Quarter: CA1 is driven by ECin, not by CA3 recall
	// (which is not really active yet anyway)
	ca1FmECin.WtScale.Abs = 1
	ca1FmCa3.WtScale.Abs = 0

	dgwtscale := ca3FmDg.WtScale.Rel
	ca3FmDg.WtScale.Rel = 0 // turn off DG input to CA3 in first quarter

	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train)
					}
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train)
					}
				}
			}
		}
		switch qtr + 1 {
		case 1: // Second, Third Quarters: CA1 is driven by CA3 recall
			ca1FmECin.WtScale.Abs = 0
			ca1FmCa3.WtScale.Abs = 1
			if train {
				ca3FmDg.WtScale.Rel = dgwtscale // restore after 1st quarter
			} else {
				ca3FmDg.WtScale.Rel = 1 // significantly weaker for recall
			}
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
		case 3: // Fourth Quarter: CA1 back to ECin drive only
			ca1FmECin.WtScale.Abs = 1
			ca1FmCa3.WtScale.Abs = 0
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

			if train { // clamp ECout from ECin
				ecin.UnitVals(&ss.TmpVals, "Act")
				ecout.ApplyExt1D32(ss.TmpVals)
			} else {
				ecin.UnitVals(&ss.TmpVals, "Act")
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		if qtr+1 == 3 {
			ss.MemStats(train) // must come after QuarterFinal
		}
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView(train)
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train)
				}
			}
		}
	}

	ca3FmDg.WtScale.Rel = dgwtscale // restore
	ca1FmCa3.WtScale.Abs = 1

	if train {
		ss.Net.DWt()
	}

	if train {
		ss.Net.WtFmDWt()
	}

	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
	if !train {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

func (ss *Sim) AlphaCycSleep(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.

	ca1 := ss.Net.LayerByName("CA1").(leabra.LeabraLayer).AsLeabra()
	ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	dg := ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()

	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	ecout := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	cortex := ss.Net.LayerByName("Cortex").(leabra.LeabraLayer).AsLeabra()
	letter := ss.Net.LayerByName("Letter").(leabra.LeabraLayer).AsLeabra()
	ca1FmECin := ca1.RcvPrjns.SendName("ECin").(*hip.EcCa1Prjn)
	ca1FmCa3 := ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn)
	ECinFMInput := ecin.RcvPrjns.SendName("Input").(*leabra.Prjn)
	ECinFMECout := ecin.RcvPrjns.SendName("ECout").(*leabra.Prjn)
	OutputFMCortex := out.RcvPrjns.SendName("Cortex").(*leabra.Prjn)
	OutputFMLetter := out.RcvPrjns.SendName("Letter").(*leabra.Prjn)
	OutputFMECin := out.RcvPrjns.SendName("ECout").(*leabra.Prjn)
	CortexFMOutput := cortex.RcvPrjns.SendName("Output").(*leabra.Prjn)
	ECinFMOutput := ecin.RcvPrjns.SendName("Output").(*leabra.Prjn)
	LetterFMOutput := letter.RcvPrjns.SendName("Output").(*leabra.Prjn)
	CortexFMin := cortex.RcvPrjns.SendName("Input").(*leabra.Prjn)
	ca3FmECin := ca3.RcvPrjns.SendName("ECin").(*hip.EcCa1Prjn)
	dgFmECin := dg.RcvPrjns.SendName("ECin").(*hip.CHLPrjn)
	letterhid := ss.Net.LayerByName("Letter").(leabra.LeabraLayer).AsLeabra()
	ca3Fmca3 := ca3.RcvPrjns.SendName("CA3").(*hip.EcCa1Prjn)
	ca1Fmca3 := ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn)
	ECoutFMca1 := ecout.RcvPrjns.SendName("CA1").(*hip.EcCa1Prjn)
	ca1FMECout := ca1.RcvPrjns.SendName("ECout").(*hip.EcCa1Prjn)
	ca3FmDg := ca3.RcvPrjns.SendName("DG").(leabra.LeabraPrjn).AsLeabra()

	// semantic learn switch
	sem := ss.Net.LayerByName("Semantic").(leabra.LeabraLayer).AsLeabra()
	ECinFMSemantic := ecin.RcvPrjns.SendName("Semantic").(*leabra.Prjn)
	SemanticFMInput := sem.RcvPrjns.SendName("Input").(*leabra.Prjn)
	SemanticFMECin := sem.RcvPrjns.SendName("ECin").(*leabra.Prjn)
	ECinFMSemantic.Learn.Learn = false
	SemanticFMInput.Learn.Learn = false
	SemanticFMECin.Learn.Learn = false
	ECinFMSemantic.WtScale.Rel = 2
	ECinFMInput.WtScale.Rel = 2
	//ecin.Inhib.Layer.On = false
	ECinFMECout.WtScale.Rel = 0.1
	ECinFMOutput.WtScale.Abs = 1
	ECinFMOutput.WtScale.Rel = 1
	//ca3Fmca3.CHL.On = true
	//ca1Fmca3.CHL.On = true
	//dgFmECin.CHL.On = true
	//ca3FmECin.CHL.On = true
	//ECinFMSemantic.Learn.Learn = false
	LetterFMOutput.Learn.Learn = true
	CortexFMOutput.Learn.Learn = true
	OutputFMCortex.Learn.Learn = true
	OutputFMLetter.Learn.Learn = true
	ca1FMECout.Learn.Learn = true
	ca3Fmca3.Learn.Learn = true
	ca1Fmca3.Learn.Learn = true
	ECoutFMca1.Learn.Learn = true
	ca3FmECin.Learn.Learn = true
	dgFmECin.Learn.Learn = true
	ca1FmECin.Learn.Learn = true
	cortex.Off = false
	OutputFMECin.Off = false
	letterhid.Off = false
	ECinFMInput.Off = false
	ECinFMECout.Off = false
	OutputFMLetter.Off = false
	OutputFMCortex.Off = false
	CortexFMOutput.Off = false
	CortexFMin.Off = false
	ca1FmECin.Off = false
	ca3FmECin.Off = false
	dgFmECin.Off = false
	ca1.Off = false
	ca3.Off = false
	dg.Off = false
	ecout.Off = false
	//dgFmECin.WtScale.Abs = 1
	//ca3FmECin.WtScale.Abs = 1

	//ca1.SetType(emer.Hidden)
	//ca3.SetType(emer.Hidden)
	//dg.SetType(emer.Hidden)
	dg.SetType(emer.Hidden)
	ca1.SetType(emer.Hidden)
	ca3.SetType(emer.Hidden)

	if train {
		ecin.SetType(emer.Hidden)
		ecout.SetType(emer.Target) // clamp a plus phase during testing
	} else {
		// don't clamp
		ecout.SetType(emer.Compare)
		//ecin.UpdateExtFlags() // call this after updating type
	}
	ecout.UpdateExtFlags() // call this after updating type

	if train {
		OutputFMECin.Off = true
	}
	// First Quarter: CA1 is driven by ECin, not by CA3 recall
	// (which is not really active yet anyway)
	ca1FmECin.WtScale.Abs = 1
	ca1FmCa3.WtScale.Abs = 0
	dgwtscale := ca3FmDg.WtScale.Rel
	ca3FmDg.WtScale.Rel = 0 // turn off DG input to CA3 in first quarter
	ss.Net.GScaleFmAvgAct() // update computed scaling factors
	ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train)
					}
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train)
					}
				}
			}
		}
		switch qtr + 1 {
		case 1: // Second, Third Quarters: CA1 is driven by CA3 recall
			ca1FmECin.WtScale.Abs = 0
			ca1FmCa3.WtScale.Abs = 1
			if train {
				ca3FmDg.WtScale.Rel = dgwtscale // restore after 1st quarter
			} else {
				ca3FmDg.WtScale.Rel = 1 // significantly weaker for recall
			}
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
		case 3: // Fourth Quarter: CA1 back to ECin drive only
			ca1FmECin.WtScale.Abs = 1
			ca1FmCa3.WtScale.Abs = 0
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

			if train { // clamp ECout from ECin
				ecout.UnitVals(&ss.TmpVals, "Act")
				out.ApplyExt1D32(ss.TmpVals)
				ecin.UnitVals(&ss.TmpVals, "Act")
				ecout.ApplyExt1D32(ss.TmpVals)
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		if qtr+1 == 3 {
			ss.MemStats(train) // must come after QuarterFinal
		}
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView(train)
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train)
				}
			}
		}
	}
	ca3FmDg.WtScale.Rel = dgwtscale // restore
	ca1FmCa3.WtScale.Abs = 1

	if train {
		ss.Net.DWt()
	}

	if train {
		ss.Net.WtFmDWt()
	}

	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
	if !train {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

func (ss *Sim) AlphaCycDaydream(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.

	ca1 := ss.Net.LayerByName("CA1").(leabra.LeabraLayer).AsLeabra()
	ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	dg := ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()

	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	ecout := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	cortex := ss.Net.LayerByName("Cortex").(leabra.LeabraLayer).AsLeabra()
	letter := ss.Net.LayerByName("Letter").(leabra.LeabraLayer).AsLeabra()
	ca1FmECin := ca1.RcvPrjns.SendName("ECin").(*hip.EcCa1Prjn)
	ca1FmCa3 := ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn)
	ECinFMInput := ecin.RcvPrjns.SendName("Input").(*leabra.Prjn)
	ECinFMECout := ecin.RcvPrjns.SendName("ECout").(*leabra.Prjn)
	OutputFMCortex := out.RcvPrjns.SendName("Cortex").(*leabra.Prjn)
	OutputFMLetter := out.RcvPrjns.SendName("Letter").(*leabra.Prjn)
	OutputFMECin := out.RcvPrjns.SendName("ECout").(*leabra.Prjn)
	CortexFMOutput := cortex.RcvPrjns.SendName("Output").(*leabra.Prjn)
	ECinFMOutput := ecin.RcvPrjns.SendName("Output").(*leabra.Prjn)
	LetterFMOutput := letter.RcvPrjns.SendName("Output").(*leabra.Prjn)
	CortexFMin := cortex.RcvPrjns.SendName("Input").(*leabra.Prjn)
	ca3FmECin := ca3.RcvPrjns.SendName("ECin").(*hip.EcCa1Prjn)
	dgFmECin := dg.RcvPrjns.SendName("ECin").(*hip.CHLPrjn)
	letterhid := ss.Net.LayerByName("Letter").(leabra.LeabraLayer).AsLeabra()
	ca3Fmca3 := ca3.RcvPrjns.SendName("CA3").(*hip.EcCa1Prjn)
	ca1Fmca3 := ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn)
	ECoutFMca1 := ecout.RcvPrjns.SendName("CA1").(*hip.EcCa1Prjn)
	ca1FMECout := ca1.RcvPrjns.SendName("ECout").(*hip.EcCa1Prjn)

	// semantic learn switch
	sem := ss.Net.LayerByName("Semantic").(leabra.LeabraLayer).AsLeabra()
	ECinFMSemantic := ecin.RcvPrjns.SendName("Semantic").(*leabra.Prjn)
	SemanticFMInput := sem.RcvPrjns.SendName("Input").(*leabra.Prjn)
	SemanticFMECin := sem.RcvPrjns.SendName("ECin").(*leabra.Prjn)
	ECinFMSemantic.Learn.Learn = false
	SemanticFMInput.Learn.Learn = false
	SemanticFMECin.Learn.Learn = false
	ECinFMSemantic.WtScale.Rel = float32(ss.SemWake) / 100
	ECinFMInput.WtScale.Rel = 2
	//ecin.Inhib.Layer.On = false
	ECinFMECout.WtScale.Rel = 0.1
	ECinFMOutput.WtScale.Abs = 1
	ECinFMOutput.WtScale.Rel = float32(ss.Outecback)
	//ca3Fmca3.CHL.On = true
	//ca1Fmca3.CHL.On = true
	//dgFmECin.CHL.On = true
	//ca3FmECin.CHL.On = true
	//ECinFMSemantic.Learn.Learn = false
	LetterFMOutput.Learn.Learn = true
	CortexFMOutput.Learn.Learn = true
	OutputFMCortex.Learn.Learn = true
	OutputFMLetter.Learn.Learn = true
	ca1FMECout.Learn.Learn = true
	ca3Fmca3.Learn.Learn = true
	ca1Fmca3.Learn.Learn = true
	ECoutFMca1.Learn.Learn = true
	ca3FmECin.Learn.Learn = true
	dgFmECin.Learn.Learn = true
	ca1FmECin.Learn.Learn = true
	cortex.Off = false
	OutputFMECin.Off = false
	letterhid.Off = false
	ECinFMInput.Off = false
	ECinFMECout.Off = false
	OutputFMLetter.Off = false
	OutputFMCortex.Off = false
	CortexFMOutput.Off = false
	CortexFMin.Off = false
	ca1FmECin.Off = false
	ca3FmECin.Off = false
	dgFmECin.Off = false
	ca1.Off = false
	ca3.Off = false
	dg.Off = false
	ecout.Off = false
	//dgFmECin.WtScale.Abs = 1
	//ca3FmECin.WtScale.Abs = 1

	//ca1.SetType(emer.Hidden)
	//ca3.SetType(emer.Hidden)
	//dg.SetType(emer.Hidden)
	dg.SetType(emer.Hidden)
	ca1.SetType(emer.Hidden)
	ca3.SetType(emer.Hidden)

	if train {
		ecin.SetType(emer.Hidden)
		ecout.SetType(emer.Target) // clamp a plus phase during testing
	} else {
		// don't clamp
		ecout.SetType(emer.Compare)
		//ecin.UpdateExtFlags() // call this after updating type
	}
	ecout.UpdateExtFlags() // call this after updating type

	if train {
		OutputFMECin.Off = true
	}
	// First Quarter: CA1 is driven by ECin, not by CA3 recall
	// (which is not really active yet anyway)
	ca1FmECin.WtScale.Abs = 1
	ca1FmCa3.WtScale.Abs = 0
	ss.Net.GScaleFmAvgAct() // update computed scaling factors
	ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train)
					}
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train)
					}
				}
			}
		}
		switch qtr + 1 {
		case 1: // Second, Third Quarters: CA1 is driven by CA3 recall
			ca1FmECin.WtScale.Abs = 0
			ca1FmCa3.WtScale.Abs = 1
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
		case 3: // Fourth Quarter: CA1 back to ECin drive only
			ca1FmECin.WtScale.Abs = 1
			ca1FmCa3.WtScale.Abs = 0
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

			if train { // clamp ECout from ECin
				ecout.UnitVals(&ss.TmpVals, "Act")
				out.ApplyExt1D32(ss.TmpVals)
				ecin.UnitVals(&ss.TmpVals, "Act")
				ecout.ApplyExt1D32(ss.TmpVals)
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		if qtr+1 == 3 {
			ss.MemStats(train) // must come after QuarterFinal
		}
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView(train)
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train)
				}
			}
		}
	}

	if train {
		ss.Net.DWt()
	}

	if train {
		ss.Net.WtFmDWt()
	}

	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
	if !train {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

func (ss *Sim) AlphaCycHipPureRet(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.

	ca1 := ss.Net.LayerByName("CA1").(leabra.LeabraLayer).AsLeabra()
	ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	dg := ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	ecout := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	cortex := ss.Net.LayerByName("Cortex").(leabra.LeabraLayer).AsLeabra()
	letter := ss.Net.LayerByName("Letter").(leabra.LeabraLayer).AsLeabra()
	ca1FmECin := ca1.RcvPrjns.SendName("ECin").(*hip.EcCa1Prjn)
	ca1FmCa3 := ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn)
	ECinFMInput := ecin.RcvPrjns.SendName("Input").(*leabra.Prjn)
	ECinFMECout := ecin.RcvPrjns.SendName("ECout").(*leabra.Prjn)
	OutputFMCortex := out.RcvPrjns.SendName("Cortex").(*leabra.Prjn)
	OutputFMLetter := out.RcvPrjns.SendName("Letter").(*leabra.Prjn)
	OutputFMECin := out.RcvPrjns.SendName("ECout").(*leabra.Prjn)
	//ECinFMSemantic := ecin.RcvPrjns.SendName("Semantic").(*leabra.Prjn)
	ECinFMOutput := ecin.RcvPrjns.SendName("Output").(*leabra.Prjn)
	CortexFMOutput := cortex.RcvPrjns.SendName("Output").(*leabra.Prjn)
	LetterFMOutput := letter.RcvPrjns.SendName("Output").(*leabra.Prjn)
	CortexFMin := cortex.RcvPrjns.SendName("Input").(*leabra.Prjn)
	ca3FmECin := ca3.RcvPrjns.SendName("ECin").(*hip.EcCa1Prjn)
	dgFmECin := dg.RcvPrjns.SendName("ECin").(*hip.CHLPrjn)
	letterhid := ss.Net.LayerByName("Letter").(leabra.LeabraLayer).AsLeabra()
	ca3Fmca3 := ca3.RcvPrjns.SendName("CA3").(*hip.EcCa1Prjn)
	ca1Fmca3 := ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn)
	ECoutFMca1 := ecout.RcvPrjns.SendName("CA1").(*hip.EcCa1Prjn)
	ca1FMECout := ca1.RcvPrjns.SendName("ECout").(*hip.EcCa1Prjn)
	ca3FmDg := ca3.RcvPrjns.SendName("DG").(leabra.LeabraPrjn).AsLeabra()
	//ecin.Inhib.Layer.On = false
	ECinFMECout.WtScale.Rel = 0.5
	ECinFMOutput.WtScale.Abs = 0
	ECinFMOutput.WtScale.Rel = float32(ss.Outecback)
	//ca3Fmca3.CHL.On = true
	//ca1Fmca3.CHL.On = true
	//dgFmECin.CHL.On = true
	//ca3FmECin.CHL.On = true
	//ECinFMSemantic.Learn.Learn = false
	LetterFMOutput.Learn.Learn = false
	CortexFMOutput.Learn.Learn = false
	OutputFMCortex.Learn.Learn = false
	OutputFMLetter.Learn.Learn = false
	ca1FMECout.Learn.Learn = true
	ca3Fmca3.Learn.Learn = true
	ca1Fmca3.Learn.Learn = true
	ECoutFMca1.Learn.Learn = true
	ca3FmECin.Learn.Learn = true
	dgFmECin.Learn.Learn = true
	ca1FmECin.Learn.Learn = true
	cortex.Off = true
	letterhid.Off = true
	ECinFMInput.Off = false
	ECinFMECout.Off = false
	OutputFMLetter.Off = true
	OutputFMECin.Off = false
	OutputFMCortex.Off = true
	CortexFMOutput.Off = true
	CortexFMin.Off = true
	ca1FmECin.Off = false
	ca3FmECin.Off = false
	dgFmECin.Off = false
	ca1.Off = false
	ca3.Off = false
	dg.Off = false
	ecout.Off = false
	//dgFmECin.WtScale.Abs = 1
	//ca3FmECin.WtScale.Abs = 1

	//ca1.SetType(emer.Hidden)
	//ca3.SetType(emer.Hidden)
	//dg.SetType(emer.Hidden)
	dg.SetType(emer.Hidden)
	ca1.SetType(emer.Hidden)
	ca3.SetType(emer.Hidden)

	if train {
		ecin.SetType(emer.Hidden)
		ecout.SetType(emer.Target) // clamp a plus phase during testing
	} else {
		// don't clamp
		ecout.SetType(emer.Compare)
		//ecin.UpdateExtFlags() // call this after updating type
	}
	ecout.UpdateExtFlags() // call this after updating type

	// First Quarter: CA1 is driven by ECin, not by CA3 recall
	// (which is not really active yet anyway)
	ca1FmECin.WtScale.Abs = 0
	ca1FmCa3.WtScale.Abs = 1
	ca3FmDg.WtScale.Rel = 1 //

	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train)
					}
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train)
					}
				}
			}
		}
		switch qtr + 1 {

		case 3: // Fourth Quarter: CA1 back to ECin drive only

			if train { // clamp ECout from ECin
				ecin.UnitVals(&ss.TmpVals, "Act")
				ecout.ApplyExt1D32(ss.TmpVals)
			} else {
				ecin.UnitVals(&ss.TmpVals, "Act")
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		if qtr+1 == 3 {
			ss.MemStats(train) // must come after QuarterFinal
		}
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView(train)
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train)
				}
			}
		}
	}

	if train {
		ss.Net.DWt()
	}

	if train {
		ss.Net.WtFmDWt()
	}

	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
	if !train {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

//Regular 3 layer training for the simplify cortex loop

func (ss *Sim) AlphaCycSem(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()

	//if train {
	//	ecin.WtFmDWt()
	//}

	//input := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	ca1 := ss.Net.LayerByName("CA1").(leabra.LeabraLayer).AsLeabra()
	ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	dg := ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	//input := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	ecout := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	cortex := ss.Net.LayerByName("Cortex").(leabra.LeabraLayer).AsLeabra()
	letterhid := ss.Net.LayerByName("Letter").(leabra.LeabraLayer).AsLeabra()
	ca1FmECin := ca1.RcvPrjns.SendName("ECin").(*hip.EcCa1Prjn)
	//	ca1FmCa3 := ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn)
	ECinFMInput := ecin.RcvPrjns.SendName("Input").(*leabra.Prjn)
	ECinFMECout := ecin.RcvPrjns.SendName("ECout").(*leabra.Prjn)
	OutputFMCortex := out.RcvPrjns.SendName("Cortex").(*leabra.Prjn)
	CortexFMOutput := cortex.RcvPrjns.SendName("Output").(*leabra.Prjn)
	CortexFMin := cortex.RcvPrjns.SendName("Input").(*leabra.Prjn)
	ca3FmECin := ca3.RcvPrjns.SendName("ECin").(*hip.EcCa1Prjn)
	dgFmECin := dg.RcvPrjns.SendName("ECin").(*hip.CHLPrjn)
	ca3Fmca3 := ca3.RcvPrjns.SendName("CA3").(*hip.EcCa1Prjn)
	ca1Fmca3 := ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn)
	ECoutFMca1 := ecout.RcvPrjns.SendName("CA1").(*hip.EcCa1Prjn)
	ca1FMECout := ca1.RcvPrjns.SendName("ECout").(*hip.EcCa1Prjn)
	letter := ss.Net.LayerByName("Letter").(leabra.LeabraLayer).AsLeabra()
	LetterFMOutput := letter.RcvPrjns.SendName("Output").(*leabra.Prjn)
	OutputFMLetter := out.RcvPrjns.SendName("Letter").(*leabra.Prjn)

	sem := ss.Net.LayerByName("Semantic").(leabra.LeabraLayer).AsLeabra()
	ECinFMSemantic := ecin.RcvPrjns.SendName("Semantic").(*leabra.Prjn)
	SemanticFMInput := sem.RcvPrjns.SendName("Input").(*leabra.Prjn)
	SemanticFMECin := sem.RcvPrjns.SendName("ECin").(*leabra.Prjn)
	ECinFMSemantic.Learn.Learn = true
	SemanticFMInput.Learn.Learn = true
	SemanticFMECin.Learn.Learn = true
	ECinFMSemantic.WtScale.Rel = float32(ss.Outecback)

	LetterFMOutput.Learn.Learn = false
	CortexFMOutput.Learn.Learn = false
	OutputFMCortex.Learn.Learn = false
	OutputFMLetter.Learn.Learn = false
	ca1FMECout.Learn.Learn = false
	ca3Fmca3.Learn.Learn = false
	ca1Fmca3.Learn.Learn = false
	ECoutFMca1.Learn.Learn = false
	OutputFMLetter.Off = false
	ca3FmECin.Learn.Learn = false
	dgFmECin.Learn.Learn = false
	ca1FmECin.Learn.Learn = false
	cortex.Off = true
	letterhid.Off = true
	ECinFMInput.Off = true
	ECinFMECout.Off = false
	OutputFMCortex.Off = false
	CortexFMOutput.Off = false
	CortexFMin.Off = false
	ca1FmECin.Off = false
	ca3FmECin.Off = false
	dgFmECin.Off = false
	ca1.Off = true
	ca3.Off = true
	dg.Off = true
	ecout.Off = true

	//if train {
	//	ecout.SetType(emer.Target) // clamp a plus phase during testing
	//} else {
	//	ecout.SetType(emer.Compare) // don't clamp
	//}
	//ecout.UpdateExtFlags() // call this after updating type

	// First Quarter: CA1 is driven by ECin, not by CA3 recall
	// (which is not really active yet anyway)

	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train)
					}
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train)
					}
				}
			}
		}

		ss.Net.QuarterFinal(&ss.Time)
		if qtr+1 == 3 {
			ss.MemStats(train) // must come after QuarterFinal
		}
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView(train)
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train)
				}
			}
		}
	}

	if train {
		ss.Net.DWt()
	}

	if train {
		ss.Net.WtFmDWt()
	}

	//if train {
	//	ecin.DWt()
	//}

	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
	if !train {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

func (ss *Sim) AlphaCycCor(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()

	//if train {
	//	ecin.WtFmDWt()
	//}

	//input := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	ca1 := ss.Net.LayerByName("CA1").(leabra.LeabraLayer).AsLeabra()
	ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	dg := ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	//input := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	ecout := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	cortex := ss.Net.LayerByName("Cortex").(leabra.LeabraLayer).AsLeabra()
	letterhid := ss.Net.LayerByName("Letter").(leabra.LeabraLayer).AsLeabra()
	ca1FmECin := ca1.RcvPrjns.SendName("ECin").(*hip.EcCa1Prjn)
	//	ca1FmCa3 := ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn)
	ECinFMInput := ecin.RcvPrjns.SendName("Input").(*leabra.Prjn)
	ECinFMECout := ecin.RcvPrjns.SendName("ECout").(*leabra.Prjn)
	OutputFMCortex := out.RcvPrjns.SendName("Cortex").(*leabra.Prjn)
	CortexFMOutput := cortex.RcvPrjns.SendName("Output").(*leabra.Prjn)
	CortexFMin := cortex.RcvPrjns.SendName("Input").(*leabra.Prjn)
	ca3FmECin := ca3.RcvPrjns.SendName("ECin").(*hip.EcCa1Prjn)
	dgFmECin := dg.RcvPrjns.SendName("ECin").(*hip.CHLPrjn)
	ca3Fmca3 := ca3.RcvPrjns.SendName("CA3").(*hip.EcCa1Prjn)
	ca1Fmca3 := ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn)
	ECoutFMca1 := ecout.RcvPrjns.SendName("CA1").(*hip.EcCa1Prjn)
	ca1FMECout := ca1.RcvPrjns.SendName("ECout").(*hip.EcCa1Prjn)
	letter := ss.Net.LayerByName("Letter").(leabra.LeabraLayer).AsLeabra()
	LetterFMOutput := letter.RcvPrjns.SendName("Output").(*leabra.Prjn)
	OutputFMLetter := out.RcvPrjns.SendName("Letter").(*leabra.Prjn)

	sem := ss.Net.LayerByName("Semantic").(leabra.LeabraLayer).AsLeabra()
	ECinFMSemantic := ecin.RcvPrjns.SendName("Semantic").(*leabra.Prjn)
	SemanticFMInput := sem.RcvPrjns.SendName("Input").(*leabra.Prjn)
	SemanticFMECin := sem.RcvPrjns.SendName("ECin").(*leabra.Prjn)
	ECinFMSemantic.Learn.Learn = false
	SemanticFMInput.Learn.Learn = false
	SemanticFMECin.Learn.Learn = false
	ECinFMSemantic.WtScale.Rel = 0.01

	LetterFMOutput.Learn.Learn = true
	CortexFMOutput.Learn.Learn = true
	OutputFMCortex.Learn.Learn = true
	OutputFMLetter.Learn.Learn = true
	ca1FMECout.Learn.Learn = false
	ca3Fmca3.Learn.Learn = false
	ca1Fmca3.Learn.Learn = false
	ECoutFMca1.Learn.Learn = false
	OutputFMLetter.Off = false
	ca3FmECin.Learn.Learn = false
	dgFmECin.Learn.Learn = false
	ca1FmECin.Learn.Learn = false
	cortex.Off = false
	letterhid.Off = false
	ECinFMInput.Off = true
	ECinFMECout.Off = true
	OutputFMCortex.Off = false
	CortexFMOutput.Off = false
	CortexFMin.Off = false
	ca1FmECin.Off = false
	ca3FmECin.Off = false
	dgFmECin.Off = false
	ca1.Off = true
	ca3.Off = true
	dg.Off = true
	ecout.Off = true

	//if train {
	//	ecout.SetType(emer.Target) // clamp a plus phase during testing
	//} else {
	//	ecout.SetType(emer.Compare) // don't clamp
	//}
	//ecout.UpdateExtFlags() // call this after updating type

	// First Quarter: CA1 is driven by ECin, not by CA3 recall
	// (which is not really active yet anyway)

	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train)
					}
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train)
					}
				}
			}
		}

		ss.Net.QuarterFinal(&ss.Time)
		if qtr+1 == 3 {
			ss.MemStats(train) // must come after QuarterFinal
		}
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView(train)
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train)
				}
			}
		}
	}

	if train {
		ss.Net.DWt()
	}

	if train {
		ss.Net.WtFmDWt()
	}

	//if train {
	//	ecin.DWt()
	//}

	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
	if !train {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

func (ss *Sim) AlphaCycFull(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	if train {
		ss.Net.WtFmDWt()
	}

	//input := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	ca1 := ss.Net.LayerByName("CA1").(leabra.LeabraLayer).AsLeabra()
	ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	dg := ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	ecout := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	cortex := ss.Net.LayerByName("Cortex").(leabra.LeabraLayer).AsLeabra()
	ca1FmECin := ca1.RcvPrjns.SendName("ECin").(*hip.EcCa1Prjn)
	//	ca1FmCa3 := ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn)
	ECinFMInput := ecin.RcvPrjns.SendName("Input").(*leabra.Prjn)
	ECinFMECout := ecin.RcvPrjns.SendName("ECout").(*leabra.Prjn)
	OutputFMCortex := out.RcvPrjns.SendName("Cortex").(*leabra.Prjn)
	CortexFMOutput := cortex.RcvPrjns.SendName("Output").(*leabra.Prjn)
	CortexFMin := cortex.RcvPrjns.SendName("Input").(*leabra.Prjn)
	ca3FmECin := ca3.RcvPrjns.SendName("ECin").(*hip.EcCa1Prjn)
	ca3Fmca3 := ca3.RcvPrjns.SendName("CA3").(*hip.EcCa1Prjn)
	ca1Fmca3 := ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn)
	dgFmECin := dg.RcvPrjns.SendName("ECin").(*hip.CHLPrjn)
	letterhid := ss.Net.LayerByName("Letter").(leabra.LeabraLayer).AsLeabra()
	ECoutFMca1 := ecout.RcvPrjns.SendName("CA1").(*hip.EcCa1Prjn)
	ca1FMECout := ca1.RcvPrjns.SendName("ECout").(*hip.EcCa1Prjn)
	letter := ss.Net.LayerByName("Letter").(leabra.LeabraLayer).AsLeabra()
	LetterFMOutput := letter.RcvPrjns.SendName("Output").(*leabra.Prjn)
	OutputFMLetter := out.RcvPrjns.SendName("Letter").(*leabra.Prjn)
	//OutputFMECout := out.RcvPrjns.SendName("ECout").(*leabra.Prjn)
	OutputFMECin := out.RcvPrjns.SendName("ECout").(*leabra.Prjn)
	ECinFMOutput := ecin.RcvPrjns.SendName("Output").(*leabra.Prjn)
	ca3FmDg := ca3.RcvPrjns.SendName("DG").(leabra.LeabraPrjn).AsLeabra()

	ECinFMOutput.WtScale.Abs = 1
	ECinFMOutput.WtScale.Rel = float32(ss.Outecback)
	//	ecin.Inhib.Layer.On = false

	// semantic learn switch
	sem := ss.Net.LayerByName("Semantic").(leabra.LeabraLayer).AsLeabra()
	ECinFMSemantic := ecin.RcvPrjns.SendName("Semantic").(*leabra.Prjn)
	SemanticFMInput := sem.RcvPrjns.SendName("Input").(*leabra.Prjn)
	SemanticFMECin := sem.RcvPrjns.SendName("ECin").(*leabra.Prjn)
	ECinFMSemantic.Learn.Learn = false
	SemanticFMInput.Learn.Learn = false
	SemanticFMECin.Learn.Learn = false
	ECinFMSemantic.WtScale.Rel = float32(ss.SemWake) / 100

	ECinFMInput.WtScale.Rel = 1
	//ecin.Inhib.Layer.On = false
	ECinFMECout.WtScale.Rel = 0.5

	LetterFMOutput.Learn.Learn = true
	CortexFMOutput.Learn.Learn = true
	OutputFMCortex.Learn.Learn = true
	OutputFMLetter.Learn.Learn = true
	ca1FMECout.Learn.Learn = false
	ca3Fmca3.Learn.Learn = false
	ca1Fmca3.Learn.Learn = false
	ECoutFMca1.Learn.Learn = false
	OutputFMLetter.Off = false
	ca3FmECin.Learn.Learn = false
	dgFmECin.Learn.Learn = false
	ca1FmECin.Learn.Learn = false
	cortex.Off = false
	letterhid.Off = false
	ECinFMInput.Off = false
	ECinFMECout.Off = false
	OutputFMECin.Off = false
	OutputFMCortex.Off = false
	CortexFMOutput.Off = false
	CortexFMin.Off = false
	ca1FmECin.Off = false
	ca3FmECin.Off = false
	dgFmECin.Off = false
	ca1.Off = false
	ca3.Off = false
	dg.Off = false
	ecout.Off = false

	if train {
		OutputFMECin.Off = true
	}
	dg.SetType(emer.Hidden)
	ca1.SetType(emer.Hidden)
	ca3.SetType(emer.Hidden)
	ecout.SetType(emer.Compare)
	ecout.UpdateExtFlags()

	if train {
		ecin.SetType(emer.Compare)
		ecin.UpdateExtFlags()
	}

	//if train {
	//	ecout.SetType(emer.Target) // clamp a plus phase during testing
	//} else {
	//	ecout.SetType(emer.Compare) // don't clamp
	//}
	//ecout.UpdateExtFlags() // call this after updating type

	// First Quarter: CA1 is driven by ECin, not by CA3 recall
	// (which is not really active yet anyway)
	ca1FmECin.WtScale.Abs = 0
	ca1Fmca3.WtScale.Abs = 1
	ca3FmDg.WtScale.Rel = 1
	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train)
					}
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train)
					}
				}
			}
		}
		switch qtr + 1 {
		case 1: // Second, Third Quarters: CA1 is driven by CA3 recall
			ca1FmECin.WtScale.Abs = 0
			ca1Fmca3.WtScale.Abs = 1
			//ss.Net.GScaleFmAvgAct() // update computed scaling factors
			//ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
		case 2: // Second, Third Quarters: CA1 is driven by CA3 recall
			ca1FmECin.WtScale.Abs = 0
			ca1Fmca3.WtScale.Abs = 1
		case 3: // Fourth Quarter: CA1 back to ECin drive only
			ca1FmECin.WtScale.Abs = 0
			ca1Fmca3.WtScale.Abs = 1
			//ss.Net.GScaleFmAvgAct() // update computed scaling factors
			//ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

			if train { // clamp ECout from ECin
				ecout.UnitVals(&ss.TmpVals, "Act")
				out.ApplyExt1D32(ss.TmpVals)
			}
		}
		//if qtr+1 == 3 {
		//	ecin.SetType(emer.Target)
		//	ecin.UpdateExtFlags()
		//	ECinFMECout.Off = false
		//}
		ss.Net.QuarterFinal(&ss.Time)
		if qtr+1 == 3 {
			ss.MemStats(train) // must come after QuarterFinal
		}
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView(train)
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train)
				}
			}
		}
	}

	if train {
		ss.Net.DWt()
	}
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
	if !train {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
} //for full model test and retrieval practice

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).

func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "ECout", "ECin"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

func (ss *Sim) ApplyInputsSem(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "ECin"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

func (ss *Sim) ApplyInputsRP(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "ECout"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

func (ss *Sim) ApplyInputsHip(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "ECout"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

func (ss *Sim) ApplyInputsCor(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "Output"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// TrainTrial runs one trial of training using TrainEnv

func (ss *Sim) TrainTrialHip() {

	if ss.NeedsNewRun {
		ss.NewRun()
	}
	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs { // done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}
	ss.ApplyInputsHip(&ss.TrainEnv)
	ss.AlphaCycHip(true) // train hip
	//ss.ApplyInputs(&ss.TrainEnv)
	//	ss.AlphaCycCortex(true) // train cortex
	ss.TrialStats(true) // accumulate
	ss.LogTrnTrl(ss.TrnTrlLog)
}

func (ss *Sim) TrainTrialCor() {

	if ss.NeedsNewRun {
		ss.NewRun()
	}
	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs { // done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}
	ss.ApplyInputsCor(&ss.TrainEnv)
	ss.AlphaCycCor(true) // train hip
	//ss.ApplyInputs(&ss.TrainEnv)
	//	ss.AlphaCycCortex(true) // train cortex
	ss.TrialStats(true) // accumulate
	ss.LogTrnTrl(ss.TrnTrlLog)
}

func (ss *Sim) TrainTrialSleep() {

	if ss.NeedsNewRun {
		ss.NewRun()
	}
	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs { // done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}
	ss.ApplyInputsHip(&ss.TrainEnv)
	ss.AlphaCycSleep(true) // train hip
	//ss.ApplyInputs(&ss.TrainEnv)
	//	ss.AlphaCycCortex(true) // train cortex
	ss.TrialStats(true) // accumulate
	ss.LogTrnTrl(ss.TrnTrlLog)
}

func (ss *Sim) TrainTrialDaydream() {

	if ss.NeedsNewRun {
		ss.NewRun()
	}
	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs { // done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}
	ss.ApplyInputsHip(&ss.TrainEnv)
	ss.AlphaCycDaydream(true) // train hip
	//ss.ApplyInputs(&ss.TrainEnv)
	//	ss.AlphaCycCortex(true) // train cortex
	ss.TrialStats(true) // accumulate
	ss.LogTrnTrl(ss.TrnTrlLog)
}

func (ss *Sim) TrainTrialSem() {

	if ss.NeedsNewRun {
		ss.NewRun()
	}
	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs { // done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}
	ecin.SetType(emer.Target)
	ss.ApplyInputsSem(&ss.TrainEnv)
	//	ss.AlphaCycHip(true) // train hip
	//ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCycSem(true) // train cortex
	ss.TrialStats(true)  // accumulate
	ss.LogTrnTrl(ss.TrnTrlLog)
}

func (ss *Sim) TrainTrialRetrievalPractice() {

	if ss.NeedsNewRun {
		ss.NewRun()
	}
	ss.TrainEnv.Step()
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	//	ecout := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs { // done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}
	//single trial retrieval practice
	ecin.SetType(emer.Hidden)
	ss.ApplyInputsRP(&ss.TrainEnv)
	ss.AlphaCycFull(true)
	ss.TrialStats(true) // !accumulate
	ss.LogTrnTrl(ss.TrnTrlLog)

	// seperate trial retrieval practice
	// ecin.SetType(emer.Hidden)
	// ss.ApplyInputsRP(&ss.TrainEnv)
	// ss.AlphaCycHipPureRet(false)
	// //	ss.ApplyInputsRP(&ss.TrainEnv)
	// ecin.SetType(emer.Target)
	// ss.AlphaCycCortexRP(true)
	// //	ss.AlphaCycFull(true) // !train
	// ss.TrialStats(true) // !accumulate
	// ss.LogTrnTrl(ss.TrnTrlLog)
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.LogRun(ss.RunLog)
	if ss.SaveWts {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %v\n", fnm)
		ss.Net.SaveWtsJSON(gi.FileName(fnm))
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.InitStats()
	ss.TrnTrlLog.SetNumRows(0)
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators
	ss.SumSSE = 0
	ss.SumAvgSSE = 0
	ss.SumCosDiff = 0
	ss.CntErr = 0
	ss.FirstZero = -1
	// clear rest just to make Sim look initialized
	ss.Mem = 0
	ss.TrgOnWasOffAll = 0
	ss.TrgOnWasOffCmp = 0
	ss.TrgOffWasOn = 0
	ss.TrlSSEB = 0
	ss.TrlAvgSSEB = 0
	ss.TrlSSEC = 0
	ss.TrlAvgSSEC = 0
	ss.EpcSSE = 0
	ss.EpcAvgSSE = 0
	ss.EpcPctErr = 0
	ss.EpcCosDiff = 0
}

// MemStats computes ActM vs. Target on ECout with binary counts
// must be called at end of 3rd quarter so that Targ values are
// for the entire full pattern as opposed to the plus-phase target
// values clamped from ECin activations

//Memstats is modified to record ecin
func (ss *Sim) MemStats(train bool) {
	//	ecout := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	nn := ecin.Shape().Len()
	trgOnWasOffAll := 0.0 // all units
	trgOnWasOffCmp := 0.0 // only those that required completion, missing in ECin
	trgOffWasOn := 0.0    // should have been off
	cmpN := 0.0           // completion target
	trgOnN := 0.0
	trgOffN := 0.0
	//this doesn't mean the true target. For A-B test, target for C pool is the target for AC. This is to measure how much C is activated in A-B
	trgOffWasOnB := 0.0 //
	trgOffWasOnC := 0.0 //
	trgOnWasOffB := 0.0 //
	trgOnWasOffC := 0.0
	trgOnNB := 0.0
	trgOffNB := 0.0
	trgOnNC := 0.0
	trgOffNC := 0.0

	for ni := 0; ni < nn; ni++ {
		actm := out.UnitVal1D("ActM", ni)
		trg := ecin.UnitVal1D("Targ", ni) // full pattern target
		inact := ecin.UnitVal1D("ActQ1", ni)
		if trg < 0.5 { // trgOff
			trgOffN += 1
			if actm > 0.5 {
				trgOffWasOn += 1
			}
		} else { // trgOn
			trgOnN += 1
			if inact < 0.5 { // missing in ECin -- completion target
				cmpN += 1
				if actm < 0.5 {
					trgOnWasOffAll += 1
					trgOnWasOffCmp += 1
				}
			} else {
				if actm < 0.5 {
					trgOnWasOffAll += 1
				}
			}
		}
	}
	//this is not a good way to do this but should work
	for ni := 2*49 - 1; ni < 3*49; ni++ {
		actm := out.UnitVal1D("ActM", ni)
		trg := ecin.UnitVal1D("Targ", ni) // full pattern target
		if trg < 0.5 {                    // trgOff
			trgOffNB += 1
			if actm > 0.5 {
				trgOffWasOnB += 1
			}
		} else { // trgOn
			trgOnNB += 1
			if actm < 0.5 {
				trgOnWasOffB += 1
			}
		}
	}

	for ni := 3*49 - 1; ni < 4*49; ni++ {
		actm := out.UnitVal1D("ActM", ni)
		trg := ecin.UnitVal1D("Targ", ni) // full pattern target
		if trg < 0.5 {                    // trgOff
			trgOffNC += 1
			if actm > 0.5 {
				trgOffWasOnC += 1
			}
		} else { // trgOn
			trgOnNC += 1
			if actm < 0.5 {
				trgOnWasOffC += 1
			}
		}
	}

	trgOnWasOffAll /= trgOnN
	trgOffWasOn /= trgOffN
	trgOnWasOffB /= trgOnNB
	trgOffWasOnB /= trgOffNB
	trgOnWasOffC /= trgOnNC
	trgOffWasOnC /= trgOffNC

	if train { // no cmp
		if trgOnWasOffAll < ss.MemThr && trgOffWasOn < ss.MemThr {
			ss.Mem = 1
		} else {
			ss.Mem = 0
		}
	} else { // test
		if cmpN > 0 { // should be
			trgOnWasOffCmp /= cmpN
			if trgOnWasOffCmp < ss.MemThr && trgOffWasOn < ss.MemThr {
				ss.Mem = 1
			} else {
				ss.Mem = 0
			}
		}
	}
	ss.TrgOnWasOffAll = trgOnWasOffAll
	ss.TrgOnWasOffCmp = trgOnWasOffCmp
	ss.TrgOffWasOn = trgOffWasOn
	ss.TrgOffWasOnB = trgOffWasOnB
	ss.TrgOnWasOffB = trgOnWasOffB
	ss.TrgOffWasOnC = trgOffWasOnC
	ss.TrgOnWasOffC = trgOnWasOffC
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) (sse, avgsse, ssec, avgssec, cosdiff float64) {
	outLay := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	ss.TrlCosDiff = float64(outLay.CosDiff.Cos)
	ss.TrlSSEB, ss.TrlAvgSSEB = outLay.MSEB(0.5) // 0.5 = per-unit tolerance -- right side of .5
	ss.TrlSSEC, ss.TrlAvgSSEC = outLay.MSEC(0.5) // 0.5 = per-unit tolerance -- right side of .5
	if accum {
		ss.SumSSE += ss.TrlSSEB
		ss.SumAvgSSE += ss.TrlAvgSSEB
		ss.SumCosDiff += ss.TrlCosDiff
		if ss.TrlSSEB != 0 {
			ss.CntErr++
		}
	}
	return
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	for {
		ss.TrainTrialSem()
		if ss.StopNow || ss.TrainEnv.Epoch.Cur != curEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrialSem()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

// Train runs the full training from this point onward

func (ss *Sim) TrainHip() {
	ss.TrainEnv.Trial.Cur = -1
	ss.StopNow = false
	for {
		ss.TrainTrialHip()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

func (ss *Sim) TrainCor() {
	ss.TrainEnv.Trial.Cur = -1
	ss.StopNow = false
	for {
		ss.TrainTrialCor()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

func (ss *Sim) TrainSleep() {
	ss.TrainEnv.Trial.Cur = -1
	ss.StopNow = false
	for {
		ss.TrainTrialSleep()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

func (ss *Sim) TrainDaydream() {
	ss.TrainEnv.Trial.Cur = -1
	ss.StopNow = false
	for {
		ss.TrainTrialDaydream()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

func (ss *Sim) TrainSem() {
	ss.TrainEnv.Trial.Cur = -1
	ss.StopNow = false
	for {
		ss.TrainTrialSem()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

func (ss *Sim) TrainRetrievalPractice() {
	ss.ConfigEnv()
	ss.TrainEnv.Trial.Cur = -1
	ss.StopNow = false
	for {
		ss.TrainTrialRetrievalPractice()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		vp.BlockUpdates()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.UnblockUpdates()
		vp.SetNeedsFullRender()
	}
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.AllOn()
	ss.Net.SaveWtsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrialSem(returnOnChg bool) {
	ss.TestEnv.Step()
	//	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView(false)
		}
		if returnOnChg {
			return
		}
	}

	//ecin.SetType(emer.Target)
	ss.ApplyInputsSem(&ss.TestEnv)

	ss.AlphaCycSem(false) // !train
	ss.TrialStats(false)  // !accumulate
	ss.LogTstTrl(ss.TstTrlLog)
}

func (ss *Sim) TestTrialHip(returnOnChg bool) {
	//	ss.Params = SavedParamsSets
	//	ss.SetParams("Network", ss.LogSetParams) // only set Network params
	ss.TestEnv.Step()
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView(false)
		}
		if returnOnChg {
			return
		}
	}
	ecin.SetType(emer.Target)
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCycHip(false) // !train
	ss.TrialStats(false)  // !accumulate
	ss.LogTstTrl(ss.TstTrlLog)
}

func (ss *Sim) TestTrialHipPure(returnOnChg bool) {
	//	ss.Params = SavedParamsSets
	//	ss.SetParams("Network", ss.LogSetParams) // only set Network params
	ss.TestEnv.Step()
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView(false)
		}
		if returnOnChg {
			return
		}
	}
	ecin.SetType(emer.Target)
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCycHipPureRet(false) // !train
	ss.TrialStats(false)         // !accumulate
	ss.LogTstTrl(ss.TstTrlLog)
}

func (ss *Sim) TestTrialFull(returnOnChg bool) {
	ss.TestEnv.Step()
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView(false)
		}
		if returnOnChg {
			return
		}
	}
	ecin.SetType(emer.Target)
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCycFull(false) // !train
	ss.TrialStats(false)   // !accumulate
	ss.LogTstTrl(ss.TstTrlLog)
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCycHip(false) // !train
	ss.TrialStats(false)  // !accumulate
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.TstTrlLog = &etable.Table{}
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	for {
		ss.TestTrialSem(true) // return on chg
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}

	// log only at very end
	ss.LogTstEpc(ss.TstEpcLog)
}

func (ss *Sim) TestAllHip() {
	ss.TstTrlLog = &etable.Table{}
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	for {
		ss.TestTrialHip(true) // return on chg
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}

	// log only at very end
	ss.LogTstEpc(ss.TstEpcLog)
}

func (ss *Sim) TestAllHipPure() {
	ss.TstTrlLog = &etable.Table{}
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	for {
		ss.TestTrialHipPure(true) // return on chg
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}

	// log only at very end
	ss.LogTstEpc(ss.TstEpcLog)
}

func (ss *Sim) TestAllFull() {
	ss.TstTrlLog = &etable.Table{}
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	for {
		ss.TestTrialFull(true) // return on chg
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}

	// log only at very end
	ss.LogTstEpc(ss.TstEpcLog)
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAllSem() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

func (ss *Sim) RunTestAllHip() {
	ss.StopNow = false
	ss.TestAllHip()
	ss.Stopped()
}

func (ss *Sim) RunTestAllHipPure() {
	ss.StopNow = false
	ss.TestAllHipPure()
	ss.Stopped()
}

func (ss *Sim) RunTestAllFull() {
	ss.StopNow = false
	ss.TestAllFull()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
func (ss *Sim) SaveTstTrial(Subject, Condition, Phase string) {
	var err error
	fnm := ss.Tag + "_" + ss.Net.Nm + "_" + Subject + "_" + Condition + "_" + Phase + ".txt"
	ss.TstTrialFile, err = os.Create(fnm)
	if err != nil {
		log.Println(err)
		ss.TstTrialFile = nil
	} else {
		fmt.Printf("Saving epoch log to: %v\n", fnm)
		defer ss.TstTrialFile.Close()
		ss.RunTestAllFull()
	}
}

func (ss *Sim) SaveHipTstTrial(Subject, Condition, Phase string) {
	var err error
	fnm := ss.Tag + "_" + ss.Net.Nm + "_" + Subject + "_" + Condition + "_" + Phase + ".txt"
	ss.TstTrialFile, err = os.Create(fnm)
	if err != nil {
		log.Println(err)
		ss.TstTrialFile = nil
	} else {
		fmt.Printf("Saving epoch log to: %v\n", fnm)
		defer ss.TstTrialFile.Close()
		ss.RunTestAllHipPure()
	}
}

//30, 10, 30, 30 showing to large sleep effect
//30，7，1，20, 1 are a good set of epoch numbers (generally)
//Dec 14 tried 30 7 21 1 6 1 not differ much from 30 7 21 1 20 1
//Dec 15 tried 30 7 21 2 12 2, generally better than 1 6 1 for sleep part
//hip should be somewhere between 5-10, rp should be ~15

func (ss *Sim) CloseUnrelated() { //30 7 20 1 19 1 works in this condition
	Condition := "CloseUn"
	for sub := 0 + ss.Group*ss.MaxSubs; sub < ss.MaxSubs+ss.Group*ss.MaxSubs; sub++ {
		Subject := strconv.Itoa(sub)
		gp := strconv.Itoa(ss.Group)
		ss.NewRndSeed()
		ss.Init()
		ss.OpenPatsSem()
		ss.SetEnv(true)
		ss.MaxEpcs = 30
		ss.TrainSem()
		ss.OpenPats()
		ss.SetEnv(false)
		ss.MaxEpcs = 4
		ss.TrainHip()
		ss.SaveHipTstTrial(Subject, Condition, "hip")
		ss.SaveTstTrial(Subject, Condition, "111")
		ss.OpenPatsSleep()
		ss.SaveTstTrial(Subject, Condition, "112")
		ss.AllOn()
		wtsname := "weights" + gp + ".wts.gz"
		ss.Net.SaveWtsJSON(gi.FileName(wtsname))
		fmt.Printf("Weight saved\n")
		ss.OpenPatsRP()
		ss.MaxEpcs = 8
		ss.SetEnv(false)
		ss.TrainRetrievalPractice()
		ss.SetEnv(false)
		ss.SaveTstTrial(Subject, Condition, "221")
		ss.OpenPatsSleepFull()
		ss.SaveTstTrial(Subject, Condition, "222")
		ss.SetEnv(false)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.OpenPatsSleep()
		ss.SetEnv(false)
		ss.MaxEpcs = 4
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "332")
		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(wtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(false)
		ss.SaveTstTrial(Subject, Condition, "112_check")
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "112s")
	}
}

func (ss *Sim) CloseRelated() {
	Condition := "CloseRe"
	for sub := 0 + ss.Group*ss.MaxSubs; sub < ss.MaxSubs+ss.Group*ss.MaxSubs; sub++ {
		Subject := strconv.Itoa(sub)
		gp := strconv.Itoa(ss.Group)
		wtsname := "weights" + gp + ".wts.gz"
		ss.NewRndSeed()
		ss.Init()
		ss.OpenPatsSem()
		ss.SetEnv(false)
		ss.MaxEpcs = 30
		ss.TrainSem()
		ss.OpenPats()
		ss.SetEnv(false)
		ss.MaxEpcs = 4
		ss.TrainHip()
		ss.SaveHipTstTrial(Subject, Condition, "hip")
		ss.SaveTstTrial(Subject, Condition, "111")
		ss.OpenPatsSleep()
		ss.SaveTstTrial(Subject, Condition, "112")
		ss.AllOn()
		ss.Net.SaveWtsJSON(gi.FileName(wtsname))
		fmt.Printf("Weight saved\n")
		ss.OpenPatsRP()
		ss.MaxEpcs = 8
		ss.SetEnv(false)
		ss.TrainRetrievalPractice()
		ss.SetEnv(false)
		ss.SaveTstTrial(Subject, Condition, "221")
		ss.OpenPatsSleepFull()
		ss.SaveTstTrial(Subject, Condition, "222")
		ss.SetEnv(false)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.OpenPatsSleep()
		ss.SetEnv(false)
		ss.MaxEpcs = 4
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "332")
		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(wtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(false)
		ss.SaveTstTrial(Subject, Condition, "112_check")
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "112s")
	}
}

func (ss *Sim) FarUnrelated() {
	Condition := "FarUn"
	for sub := 0 + ss.Group*ss.MaxSubs; sub < ss.MaxSubs+ss.Group*ss.MaxSubs; sub++ {
		Subject := strconv.Itoa(sub)
		gp := strconv.Itoa(ss.Group)
		wtsname := "weights" + gp + ".wts.gz"
		ss.NewRndSeed()
		ss.Init()
		ss.OpenPatsSem()
		ss.SetEnv(true)
		ss.MaxEpcs = 30
		ss.TrainSem()
		ss.OpenPats()
		ss.SetEnv(true)
		ss.MaxEpcs = 4
		ss.TrainHip()
		ss.SaveHipTstTrial(Subject, Condition, "hip")
		ss.SaveTstTrial(Subject, Condition, "111")
		ss.OpenPatsSleep()
		ss.SaveTstTrial(Subject, Condition, "112")
		ss.AllOn()
		ss.Net.SaveWtsJSON(gi.FileName(wtsname))
		fmt.Printf("Weight saved\n")
		ss.OpenPatsRP()
		ss.MaxEpcs = 8
		ss.SetEnv(true)
		ss.TrainRetrievalPractice()
		ss.SetEnv(true)
		ss.SaveTstTrial(Subject, Condition, "221")
		ss.OpenPatsSleepFull()
		ss.SaveTstTrial(Subject, Condition, "222")
		ss.SetEnv(true)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.OpenPatsSleep()
		ss.SetEnv(true)
		ss.MaxEpcs = 4
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "332")
		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(wtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(true)
		ss.SaveTstTrial(Subject, Condition, "112_check")
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "112s")
	}
}

func (ss *Sim) FarRelated() {
	Condition := "FarRe"
	for sub := 0 + ss.Group*ss.MaxSubs; sub < ss.MaxSubs+ss.Group*ss.MaxSubs; sub++ {
		Subject := strconv.Itoa(sub)
		gp := strconv.Itoa(ss.Group)
		wtsname := "weights" + gp + ".wts.gz"
		ss.NewRndSeed()
		ss.Init()
		ss.OpenPatsSem()
		ss.SetEnv(false)
		ss.MaxEpcs = 30
		ss.TrainSem()
		ss.OpenPats()
		ss.SetEnv(true)
		ss.MaxEpcs = 4
		ss.TrainHip()
		ss.SaveHipTstTrial(Subject, Condition, "hip")
		ss.SaveTstTrial(Subject, Condition, "111")
		ss.OpenPatsSleep()
		ss.SaveTstTrial(Subject, Condition, "112")
		ss.AllOn()
		fmt.Printf("Weight saving\n")
		ss.Net.SaveWtsJSON(gi.FileName(wtsname))
		ss.OpenPatsRP()
		ss.MaxEpcs = 8
		ss.SetEnv(true)
		ss.TrainRetrievalPractice()
		ss.SetEnv(true)
		ss.SaveTstTrial(Subject, Condition, "221")
		ss.OpenPatsSleepFull()
		ss.SetEnv(true)
		ss.SaveTstTrial(Subject, Condition, "222")
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.OpenPatsSleep()
		ss.SetEnv(true)
		ss.MaxEpcs = 4
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "332")
		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(wtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(true)
		ss.SaveTstTrial(Subject, Condition, "112_check")
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "112s")
		//ss.Init()
		//ss.NewRndSeed()
		//ss.OpenPatsSem()
		//ss.SetEnv(false)
		//ss.MaxEpcs = 30
		//ss.TrainSem()
		//ss.OpenPats()
		//ss.SetEnv(true)
		//ss.MaxEpcs = 5
		//ss.TrainHip()
		//ss.SaveHipTstTrial(Subject, Condition, "hip")
		//ss.SaveTstTrial(Subject, Condition, "111")
		//ss.OpenPatsSleep()
		//ss.SaveTstTrial(Subject, Condition, "112")
		//ss.OpenPatsRP()
		//ss.MaxEpcs = 10
		//ss.SetEnv(true)
		//ss.TrainRetrievalPractice()
		//ss.SetEnv(true)
		//ss.SaveTstTrial(Subject, Condition, "221")
		//ss.OpenPatsSleep()
		//ss.SaveTstTrial(Subject, Condition, "222")
		//ss.SetEnv(true)
		//ss.MaxEpcs = 10
		//ss.TrainSleep()
		//ss.SaveTstTrial(Subject, Condition, "332")
	}
}

func (ss *Sim) CloseUnrelatedDaydream() { //30 7 20 1 19 1 works in this condition
	Condition := "CloseUn"
	for sub := 0 + ss.Group*ss.MaxSubs; sub < ss.MaxSubs+ss.Group*ss.MaxSubs; sub++ {
		Subject := strconv.Itoa(sub)
		gp := strconv.Itoa(ss.Group)
		ss.NewRndSeed()
		ss.Init()
		ss.OpenPatsSem()
		ss.SetEnv(true)

		ss.MaxEpcs = 30
		ss.TrainSem()
		ss.OpenPats()
		ss.SetEnv(false)
		ss.MaxEpcs = 8
		ss.TrainHip()
		ss.SaveHipTstTrial(Subject, Condition, "hip")
		ss.SaveTstTrial(Subject, Condition, "111")
		ss.OpenPatsSleep()
		ss.SaveTstTrial(Subject, Condition, "112")
		ss.AllOn()
		hipwtsname := "weightship" + gp + ".wts.gz"
		ss.Net.SaveWtsJSON(gi.FileName(hipwtsname))
		fmt.Printf("Weight saved\n")
		ss.OpenPatsRP()
		ss.MaxEpcs = 16
		ss.SetEnv(false)
		ss.TrainRetrievalPractice()
		ss.SetEnv(false)
		ss.SaveTstTrial(Subject, Condition, "221")
		ss.OpenPatsSleepFull()
		ss.SaveTstTrial(Subject, Condition, "222")
		ss.AllOn()
		rpwtsname := "weightsrp" + gp + ".wts.gz"
		ss.Net.SaveWtsJSON(gi.FileName(rpwtsname))
		ss.SetEnv(false)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.OpenPatsSleep()
		ss.SetEnv(false)
		ss.MaxEpcs = 7
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "222s")
		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(hipwtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(false)
		ss.SaveTstTrial(Subject, Condition, "112c")
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "112s")

		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(hipwtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(false)
		ss.SaveTstTrial(Subject, Condition, "112cc")
		ss.MaxEpcs = 1
		ss.TrainDaydream()
		ss.SaveTstTrial(Subject, Condition, "112d")

		ss.OpenPatsSleepFull()
		ss.SetEnv(false)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "112ds")

		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(rpwtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(false)
		ss.SaveTstTrial(Subject, Condition, "222c")
		ss.MaxEpcs = 1
		ss.TrainDaydream()
		ss.OpenPatsSleep()
		ss.SetEnv(false)
		ss.MaxEpcs = 7
		ss.TrainDaydream()
		ss.SaveTstTrial(Subject, Condition, "222d")

		ss.OpenPatsSleepFull()
		ss.SetEnv(false)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.OpenPatsSleep()
		ss.SetEnv(false)
		ss.MaxEpcs = 7
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "222ds")
	}
}

func (ss *Sim) CloseRelatedDaydream() {
	Condition := "CloseRe"
	for sub := 0 + ss.Group*ss.MaxSubs; sub < ss.MaxSubs+ss.Group*ss.MaxSubs; sub++ {
		Subject := strconv.Itoa(sub)
		gp := strconv.Itoa(ss.Group)
		ss.NewRndSeed()
		ss.Init()
		ss.OpenPatsSem()
		ss.SetEnv(false)
		ss.MaxEpcs = 30
		ss.TrainSem()
		ss.OpenPats()
		ss.SetEnv(false)
		ss.MaxEpcs = 8
		ss.TrainHip()
		ss.SaveHipTstTrial(Subject, Condition, "hip")
		ss.SaveTstTrial(Subject, Condition, "111")
		ss.OpenPatsSleep()
		ss.SaveTstTrial(Subject, Condition, "112")
		ss.AllOn()
		hipwtsname := "weightship" + gp + ".wts.gz"
		ss.Net.SaveWtsJSON(gi.FileName(hipwtsname))
		fmt.Printf("Weight saved\n")
		ss.OpenPatsRP()
		ss.MaxEpcs = 16
		ss.SetEnv(false)
		ss.TrainRetrievalPractice()
		ss.SetEnv(false)
		ss.SaveTstTrial(Subject, Condition, "221")
		ss.OpenPatsSleepFull()
		ss.SaveTstTrial(Subject, Condition, "222")
		ss.AllOn()
		rpwtsname := "weightsrp" + gp + ".wts.gz"
		ss.Net.SaveWtsJSON(gi.FileName(rpwtsname))
		ss.SetEnv(false)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.OpenPatsSleep()
		ss.SetEnv(false)
		ss.MaxEpcs = 7
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "222s")
		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(hipwtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(false)
		ss.SaveTstTrial(Subject, Condition, "112c")
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "112s")

		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(hipwtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(false)
		ss.SaveTstTrial(Subject, Condition, "112cc")
		ss.MaxEpcs = 1
		ss.TrainDaydream()
		ss.SaveTstTrial(Subject, Condition, "112d")

		ss.OpenPatsSleepFull()
		ss.SetEnv(false)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "112ds")

		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(rpwtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(false)
		ss.SaveTstTrial(Subject, Condition, "222c")
		ss.MaxEpcs = 1
		ss.TrainDaydream()
		ss.OpenPatsSleep()
		ss.SetEnv(false)
		ss.MaxEpcs = 7
		ss.TrainDaydream()
		ss.SaveTstTrial(Subject, Condition, "222d")

		ss.OpenPatsSleepFull()
		ss.SetEnv(false)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.OpenPatsSleep()
		ss.SetEnv(false)
		ss.MaxEpcs = 7
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "222ds")
	}
}

func (ss *Sim) FarUnrelatedDaydream() {
	Condition := "FarUn"
	for sub := 0 + ss.Group*ss.MaxSubs; sub < ss.MaxSubs+ss.Group*ss.MaxSubs; sub++ {
		Subject := strconv.Itoa(sub)
		gp := strconv.Itoa(ss.Group)
		ss.NewRndSeed()
		ss.Init()
		ss.OpenPatsSem()
		ss.SetEnv(true)
		ss.MaxEpcs = 30
		ss.TrainSem()
		ss.OpenPats()
		ss.SetEnv(true)
		ss.MaxEpcs = 8
		ss.TrainHip()
		ss.SaveHipTstTrial(Subject, Condition, "hip")
		ss.SaveTstTrial(Subject, Condition, "111")
		ss.OpenPatsSleep()
		ss.SaveTstTrial(Subject, Condition, "112")
		ss.AllOn()
		hipwtsname := "weightship" + gp + ".wts.gz"
		ss.Net.SaveWtsJSON(gi.FileName(hipwtsname))
		fmt.Printf("Weight saved\n")
		ss.OpenPatsRP()
		ss.MaxEpcs = 16
		ss.SetEnv(true)
		ss.TrainRetrievalPractice()
		ss.SetEnv(true)
		ss.SaveTstTrial(Subject, Condition, "221")
		ss.OpenPatsSleepFull()
		ss.SaveTstTrial(Subject, Condition, "222")
		ss.AllOn()
		rpwtsname := "weightsrp" + gp + ".wts.gz"
		ss.Net.SaveWtsJSON(gi.FileName(rpwtsname))
		ss.SetEnv(true)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.OpenPatsSleep()
		ss.SetEnv(true)
		ss.MaxEpcs = 7
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "222s")
		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(hipwtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(true)
		ss.SaveTstTrial(Subject, Condition, "112c")
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "112s")

		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(hipwtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(true)
		ss.SaveTstTrial(Subject, Condition, "112cc")
		ss.MaxEpcs = 1
		ss.TrainDaydream()
		ss.SaveTstTrial(Subject, Condition, "112d")

		ss.OpenPatsSleepFull()
		ss.SetEnv(true)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "112ds")

		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(rpwtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(true)
		ss.SaveTstTrial(Subject, Condition, "222c")
		ss.MaxEpcs = 1
		ss.TrainDaydream()
		ss.OpenPatsSleep()
		ss.SetEnv(true)
		ss.MaxEpcs = 7
		ss.TrainDaydream()
		ss.SaveTstTrial(Subject, Condition, "222d")

		ss.OpenPatsSleepFull()
		ss.SetEnv(true)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.OpenPatsSleep()
		ss.SetEnv(true)
		ss.MaxEpcs = 7
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "222ds")
	}
}

func (ss *Sim) FarRelatedDaydream() {
	Condition := "FarRe"
	for sub := 0 + ss.Group*ss.MaxSubs; sub < ss.MaxSubs+ss.Group*ss.MaxSubs; sub++ {
		Subject := strconv.Itoa(sub)
		gp := strconv.Itoa(ss.Group)
		ss.NewRndSeed()
		ss.Init()
		ss.OpenPatsSem()
		ss.SetEnv(false)
		ss.MaxEpcs = 30
		ss.TrainSem()
		ss.OpenPats()
		ss.SetEnv(true)
		ss.MaxEpcs = 8
		ss.TrainHip()
		ss.SaveHipTstTrial(Subject, Condition, "hip")
		ss.SaveTstTrial(Subject, Condition, "111")
		ss.OpenPatsSleep()
		ss.SaveTstTrial(Subject, Condition, "112")
		ss.AllOn()
		hipwtsname := "weightship" + gp + ".wts.gz"
		ss.Net.SaveWtsJSON(gi.FileName(hipwtsname))
		fmt.Printf("Weight saved\n")
		ss.OpenPatsRP()
		ss.MaxEpcs = 16
		ss.SetEnv(true)
		ss.TrainRetrievalPractice()
		ss.SetEnv(true)
		ss.SaveTstTrial(Subject, Condition, "221")
		ss.OpenPatsSleepFull()
		ss.SaveTstTrial(Subject, Condition, "222")
		ss.AllOn()
		rpwtsname := "weightsrp" + gp + ".wts.gz"
		ss.Net.SaveWtsJSON(gi.FileName(rpwtsname))
		ss.SetEnv(true)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.OpenPatsSleep()
		ss.SetEnv(true)
		ss.MaxEpcs = 7
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "222s")
		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(hipwtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(true)
		ss.SaveTstTrial(Subject, Condition, "112c")
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "112s")

		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(hipwtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(true)
		ss.SaveTstTrial(Subject, Condition, "112cc")
		ss.MaxEpcs = 1
		ss.TrainDaydream()
		ss.SaveTstTrial(Subject, Condition, "112d")

		ss.OpenPatsSleepFull()
		ss.SetEnv(true)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "112ds")

		ss.AllOn()
		ss.Net.OpenWtsJSON(gi.FileName(rpwtsname))
		fmt.Printf("Weight loaded\n")
		ss.OpenPatsSleepFull()
		ss.SetEnv(true)
		ss.SaveTstTrial(Subject, Condition, "222c")
		ss.MaxEpcs = 1
		ss.TrainDaydream()
		ss.OpenPatsSleep()
		ss.SetEnv(true)
		ss.MaxEpcs = 7
		ss.TrainDaydream()
		ss.SaveTstTrial(Subject, Condition, "222d")

		ss.OpenPatsSleepFull()
		ss.SetEnv(true)
		ss.MaxEpcs = 1
		ss.TrainSleep()
		ss.OpenPatsSleep()
		ss.SetEnv(true)
		ss.MaxEpcs = 7
		ss.TrainSleep()
		ss.SaveTstTrial(Subject, Condition, "222ds")
		//ss.Init()
		//ss.NewRndSeed()
		//ss.OpenPatsSem()
		//ss.SetEnv(false)
		//ss.MaxEpcs = 30
		//ss.TrainSem()
		//ss.OpenPats()
		//ss.SetEnv(true)
		//ss.MaxEpcs = 5
		//ss.TrainHip()
		//ss.SaveHipTstTrial(Subject, Condition, "hip")
		//ss.SaveTstTrial(Subject, Condition, "111")
		//ss.OpenPatsSleep()
		//ss.SaveTstTrial(Subject, Condition, "112")
		//ss.OpenPatsRP()
		//ss.MaxEpcs = 10
		//ss.SetEnv(true)
		//ss.TrainRetrievalPractice()
		//ss.SetEnv(true)
		//ss.SaveTstTrial(Subject, Condition, "221")
		//ss.OpenPatsSleep()
		//ss.SaveTstTrial(Subject, Condition, "222")
		//ss.SetEnv(true)
		//ss.MaxEpcs = 10
		//ss.TrainSleep()
		//ss.SaveTstTrial(Subject, Condition, "332")
	}
}

func (ss *Sim) FullExp() {

	if ss.Daydream == false {
		ss.CloseUnrelated()
		ss.CloseRelated()
		ss.FarUnrelated()
		ss.FarRelated()
	} else {
		ss.CloseUnrelatedDaydream()
		ss.CloseRelatedDaydream()
		ss.FarUnrelatedDaydream()
		ss.FarRelatedDaydream()
	}

}

////////////////////////////////////////////////////////////////////////////////////
//   Params setting

// ParamsName returns name of current set of parameters
func (ss *Sim) ParamsName() string {
	if ss.ParamSet == "" {
		return "Base"
	}
	return ss.ParamSet
}

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.ParamSet != "" && ss.ParamSet != "Base" {
		err = ss.SetParamsSet(ss.ParamSet, sheet, setMsg)
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

func (ss *Sim) OpenPat(dt *etable.Table, fname, desc string) {
	err := dt.OpenCSV(gi.FileName(fname), etable.Tab)
	if err != nil {
		log.Println(err)
		return
	}
	dt.SetMetaData("name", strings.TrimSuffix(fname, ".dat"))
	dt.SetMetaData("desc", desc)
}

func (ss *Sim) OpenPats() {
	// one-time conversion from C++ patterns to Go patterns
	// patgen.ReshapeCppFile(ss.TrainClose, "Train_AB.dat", "TrainClose.dat")
	// patgen.ReshapeCppFile(ss.TrainFar, "Train_AC.dat", "TrainFar.dat")
	// patgen.ReshapeCppFile(ss.TestClose, "Test_AB.dat", "TestClose.dat")
	// patgen.ReshapeCppFile(ss.TestFar, "Test_AC.dat", "TestFar.dat")
	// patgen.ReshapeCppFile(ss.TestLure, "Lure.dat", "TestLure.dat")
	ss.OpenPat(ss.TrainClose, "close_train.txt", "Close Training Patterns")
	ss.OpenPat(ss.TrainFar, "far_train.txt", "Far Training Patterns")
	ss.OpenPat(ss.TestClose, "close_test.txt", "Close Testing Patterns")
	ss.OpenPat(ss.TestFar, "far_test.txt", "Far Testing Patterns")
}

func (ss *Sim) OpenPatsSem() {
	// one-time conversion from C++ patterns to Go patterns
	// patgen.ReshapeCppFile(ss.TrainClose, "Train_AB.dat", "TrainClose.dat")
	// patgen.ReshapeCppFile(ss.TrainFar, "Train_AC.dat", "TrainFar.dat")
	// patgen.ReshapeCppFile(ss.TestClose, "Test_AB.dat", "TestClose.dat")
	// patgen.ReshapeCppFile(ss.TestFar, "Test_AC.dat", "TestFar.dat")
	// patgen.ReshapeCppFile(ss.TestLure, "Lure.dat", "TestLure.dat")
	ss.OpenPat(ss.TrainClose, "sem_related.txt", "Close Training Patterns")
	ss.OpenPat(ss.TrainFar, "sem_unrelated.txt", "Far Training Patterns")
	ss.OpenPat(ss.TestClose, "sem_related.txt", "Close Testing Patterns")
	ss.OpenPat(ss.TestFar, "sem_unrelated.txt", "Far Testing Patterns")
}

func (ss *Sim) OpenPatsRP() {
	// one-time conversion from C++ patterns to Go patterns
	// patgen.ReshapeCppFile(ss.TrainClose, "Train_AB.dat", "TrainClose.dat")
	// patgen.ReshapeCppFile(ss.TrainFar, "Train_AC.dat", "TrainFar.dat")
	// patgen.ReshapeCppFile(ss.TestClose, "Test_AB.dat", "TestClose.dat")
	// patgen.ReshapeCppFile(ss.TestFar, "Test_AC.dat", "TestFar.dat")
	// patgen.ReshapeCppFile(ss.TestLure, "Lure.dat", "TestLure.dat")
	ss.OpenPat(ss.TrainClose, "close_rp.txt", "Close Training Patterns")
	ss.OpenPat(ss.TrainFar, "far_rp.txt", "Far Training Patterns")
	ss.OpenPat(ss.TestClose, "close_test.txt", "Close Testing Patterns")
	ss.OpenPat(ss.TestFar, "far_test.txt", "Far Testing Patterns")
}

func (ss *Sim) OpenPatsSleep() {
	// one-time conversion from C++ patterns to Go patterns
	// patgen.ReshapeCppFile(ss.TrainClose, "Train_AB.dat", "TrainClose.dat")
	// patgen.ReshapeCppFile(ss.TrainFar, "Train_AC.dat", "TrainFar.dat")
	// patgen.ReshapeCppFile(ss.TestClose, "Test_AB.dat", "TestClose.dat")
	// patgen.ReshapeCppFile(ss.TestFar, "Test_AC.dat", "TestFar.dat")
	// patgen.ReshapeCppFile(ss.TestLure, "Lure.dat", "TestLure.dat")
	ss.OpenPat(ss.TrainClose, "sleep.txt", "Close Training Patterns")
	ss.OpenPat(ss.TrainFar, "sleep.txt", "Far Training Patterns")
	ss.OpenPat(ss.TestClose, "final_test.txt", "Close Testing Patterns")
	ss.OpenPat(ss.TestFar, "final_test.txt", "Far Testing Patterns")
}

func (ss *Sim) OpenPatsSleepFull() {
	// one-time conversion from C++ patterns to Go patterns
	// patgen.ReshapeCppFile(ss.TrainClose, "Train_AB.dat", "TrainClose.dat")
	// patgen.ReshapeCppFile(ss.TrainFar, "Train_AC.dat", "TrainFar.dat")
	// patgen.ReshapeCppFile(ss.TestClose, "Test_AB.dat", "TestClose.dat")
	// patgen.ReshapeCppFile(ss.TestFar, "Test_AC.dat", "TestFar.dat")
	// patgen.ReshapeCppFile(ss.TestLure, "Lure.dat", "TestLure.dat")
	ss.OpenPat(ss.TrainClose, "sleep_full_noletter.txt", "Close Training Patterns")
	ss.OpenPat(ss.TrainFar, "sleep_full_noletter.txt", "Far Training Patterns")
	ss.OpenPat(ss.TestClose, "final_test.txt", "Close Testing Patterns")
	ss.OpenPat(ss.TestFar, "final_test.txt", "Far Testing Patterns")
}

func (ss *Sim) OpenPatsDaydream() {
	// one-time conversion from C++ patterns to Go patterns
	// patgen.ReshapeCppFile(ss.TrainClose, "Train_AB.dat", "TrainClose.dat")
	// patgen.ReshapeCppFile(ss.TrainFar, "Train_AC.dat", "TrainFar.dat")
	// patgen.ReshapeCppFile(ss.TestClose, "Test_AB.dat", "TestClose.dat")
	// patgen.ReshapeCppFile(ss.TestFar, "Test_AC.dat", "TestFar.dat")
	// patgen.ReshapeCppFile(ss.TestLure, "Lure.dat", "TestLure.dat")
	ss.OpenPat(ss.TrainClose, "Daydream.txt", "Close Training Patterns")
	ss.OpenPat(ss.TrainFar, "Daydream.txt", "Far Training Patterns")
	ss.OpenPat(ss.TestClose, "final_test.txt", "Close Testing Patterns")
	ss.OpenPat(ss.TestFar, "final_test.txt", "Far Testing Patterns")
}

func (ss *Sim) OpenPatsDaydreamFull() {
	// one-time conversion from C++ patterns to Go patterns
	// patgen.ReshapeCppFile(ss.TrainClose, "Train_AB.dat", "TrainClose.dat")
	// patgen.ReshapeCppFile(ss.TrainFar, "Train_AC.dat", "TrainFar.dat")
	// patgen.ReshapeCppFile(ss.TestClose, "Test_AB.dat", "TestClose.dat")
	// patgen.ReshapeCppFile(ss.TestFar, "Test_AC.dat", "TestFar.dat")
	// patgen.ReshapeCppFile(ss.TestLure, "Lure.dat", "TestLure.dat")
	ss.OpenPat(ss.TrainClose, "Daydream_full_close.txt", "Close Training Patterns")
	ss.OpenPat(ss.TrainFar, "Daydream_full_far.txt", "Far Training Patterns")
	ss.OpenPat(ss.TestClose, "final_test.txt", "Close Testing Patterns")
	ss.OpenPat(ss.TestFar, "final_test.txt", "Far Testing Patterns")
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	if ss.Tag != "" {
		return ss.Tag + "_" + ss.ParamsName()
	} else {
		return ss.ParamsName()
	}
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".csv"
}

//////////////////////////////////////////////
//  TrnTrlLog

// LogTrnTrl adds data from current trial to the TrnTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTrnTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Cur
	trl := ss.TrainEnv.Trial.Cur

	row := dt.Rows
	if trl == 0 { // reset at start
		row = 0
	}
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)
	dt.SetCellFloat("SSE", row, ss.TrlSSEB)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSEB)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	dt.SetCellFloat("Mem", row, ss.Mem)
	dt.SetCellFloat("TrgOnWasOff", row, ss.TrgOnWasOffAll)
	dt.SetCellFloat("TrgOffWasOn", row, ss.TrgOffWasOn)

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTrnTrlLog(dt *etable.Table) {
	// inLay := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	// outLay := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Mem", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", etensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTrnTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Train Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("Trial", false, true, 0, false, 0)
	plt.SetColParams("TrialName", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	plt.SetColParams("Mem", true, true, 0, true, 1)
	plt.SetColParams("TrgOnWasOff", true, true, 0, true, 1)
	plt.SetColParams("TrgOffWasOn", true, true, 0, true, 1)

	return plt
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv           // this is triggered by increment so use previous value
	nt := float64(ss.TrainEnv.Table.Len()) // number of trials in view

	ss.EpcSSE = ss.SumSSE / nt
	ss.SumSSE = 0
	ss.EpcAvgSSE = ss.SumAvgSSE / nt
	ss.SumAvgSSE = 0
	ss.EpcPctErr = float64(ss.CntErr) / nt
	ss.CntErr = 0
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpcCosDiff = ss.SumCosDiff / nt
	ss.SumCosDiff = 0
	if ss.FirstZero < 0 && ss.EpcPctErr == 0 {
		ss.FirstZero = epc
	}

	trlog := ss.TrnTrlLog
	tix := etable.NewIdxView(trlog)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, ss.EpcSSE)
	dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
	dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)

	dt.SetCellFloat("Mem", row, agg.Mean(tix, "Mem")[0])
	dt.SetCellFloat("TrgOnWasOff", row, agg.Mean(tix, "TrgOnWasOff")[0])
	dt.SetCellFloat("TrgOffWasOn", row, agg.Mean(tix, "TrgOffWasOn")[0])

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActAvg", row, float64(ly.Pools[0].ActAvg.ActPAvgEff))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Mem", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", etensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActAvg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", false, true, 0, true, 1)
	plt.SetColParams("PctCor", false, true, 0, true, 1)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	plt.SetColParams("Mem", true, true, 0, true, 1)         // default plot
	plt.SetColParams("TrgOnWasOff", true, true, 0, true, 1) // default plot
	plt.SetColParams("TrgOffWasOn", true, true, 0, true, 1) // default plot

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActAvg", false, true, 0, true, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	trl := ss.TestEnv.Trial.Cur
	row := dt.Rows

	if ss.TestNm == "Close" && trl == 0 { // reset at start
		row = 0
	}

	dt.SetNumRows(row + 1)
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellString("TestNm", row, ss.TestNm)
	dt.SetCellFloat("Trial", row, float64(row))
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)
	dt.SetCellFloat("SSEB", row, ss.TrlSSEB)
	dt.SetCellFloat("AvgSSEB", row, ss.TrlAvgSSEB)
	dt.SetCellFloat("SSEC", row, ss.TrlSSEC)
	dt.SetCellFloat("AvgSSEC", row, ss.TrlAvgSSEC)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)
	dt.SetCellFloat("Mem", row, ss.Mem)
	dt.SetCellFloat("TrgOnWasOff", row, ss.TrgOnWasOffCmp)
	dt.SetCellFloat("TrgOffWasOn", row, ss.TrgOffWasOn)
	dt.SetCellFloat("TrgOnWasOffB", row, ss.TrgOnWasOffB)
	dt.SetCellFloat("TrgOffWasOnB", row, ss.TrgOffWasOnB)
	dt.SetCellFloat("TrgOnWasOffC", row, ss.TrgOnWasOffC)
	dt.SetCellFloat("TrgOffWasOnC", row, ss.TrgOffWasOnC)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()

	if ss.TstTrialFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.TstTrialFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TstTrialFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	// inLay := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	// outLay := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"TestNm", etensor.STRING, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"SSEB", etensor.FLOAT64, nil, nil},
		{"AvgSSEB", etensor.FLOAT64, nil, nil},
		{"SSEC", etensor.FLOAT64, nil, nil},
		{"AvgSSEC", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Mem", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", etensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOffB", etensor.FLOAT64, nil, nil},
		{"TrgOffWasOnB", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOffC", etensor.FLOAT64, nil, nil},
		{"TrgOffWasOnC", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActM.Avg", etensor.FLOAT64, nil, nil})
	}
	// sch = append(sch, etable.Schema{
	// 	{"InAct", etensor.FLOAT64, inLay.Shp.Shp, nil},
	// 	{"OutActM", etensor.FLOAT64, outLay.Shp.Shp, nil},
	// 	{"OutActP", etensor.FLOAT64, outLay.Shp.Shp, nil},
	// }...)
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("TestNm", false, true, 0, false, 0)
	plt.SetColParams("Trial", false, true, 0, false, 0)
	plt.SetColParams("TrialName", false, true, 0, false, 0)
	plt.SetColParams("SSEB", false, true, 0, false, 0)
	plt.SetColParams("AvgSSEB", false, true, 0, false, 0)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	plt.SetColParams("Mem", true, true, 0, true, 1)
	plt.SetColParams("TrgOnWasOff", true, true, 0, true, 1)
	plt.SetColParams("TrgOffWasOn", true, true, 0, true, 1)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActM.Avg", false, true, 0, true, .5)
	}

	// plt.SetColParams("InAct", false, true, 0, true, 1)
	// plt.SetColParams("OutActM", false, true, 0, true, 1)
	// plt.SetColParams("OutActP", false, true, 0, true, 1)
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, agg.Sum(tix, "SSEB")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(tix, "AvgSSEB")[0])
	dt.SetCellFloat("PctErr", row, agg.PropIf(tix, "SSEB", func(idx int, val float64) bool {
		return val > 0
	})[0])
	dt.SetCellFloat("PctCor", row, agg.PropIf(tix, "SSEB", func(idx int, val float64) bool {
		return val == 0
	})[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])

	//trlab := etable.NewIdxView(trl)
	//trlab.Filter(func(et *etable.Table, row int) bool {
	//	return et.CellString("TestNm", row) == "Close"
	//})
	//trlac := etable.NewIdxView(trl)
	//trlac.Filter(func(et *etable.Table, row int) bool {
	//	return et.CellString("TestNm", row) == "Far"
	//})
	//
	//dt.SetCellFloat("Close Mem", row, agg.Mean(trlab, "Mem")[0])
	//dt.SetCellFloat("Close TrgOnWasOff", row, agg.Mean(trlab, "TrgOnWasOff")[0])
	//dt.SetCellFloat("Close TrgOffWasOn", row, agg.Mean(trlab, "TrgOffWasOn")[0])
	//
	//dt.SetCellFloat("Far Mem", row, agg.Mean(trlac, "Mem")[0])
	//dt.SetCellFloat("Far TrgOnWasOff", row, agg.Mean(trlac, "TrgOnWasOff")[0])
	//dt.SetCellFloat("Far TrgOffWasOn", row, agg.Mean(trlac, "TrgOffWasOn")[0])
	trix := etable.NewIdxView(trl)
	spl := split.GroupBy(trix, []string{"TestNm"})
	for _, ts := range ss.TstStatNms {
		split.Agg(spl, ts, agg.AggMean)
	}
	ss.TstStats = spl.AggsToTable(true) // no stat name
	// note: essential to use Go version of update when called from another goroutine
	ss.TstEpcPlot.GoUpdate()
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		//{"Close Mem", etensor.FLOAT64, nil, nil},
		//		//{"Close TrgOnWasOff", etensor.FLOAT64, nil, nil},
		//		//{"Close TrgOffWasOn", etensor.FLOAT64, nil, nil},
		//		//{"Far Mem", etensor.FLOAT64, nil, nil},
		//		//{"Far TrgOnWasOff", etensor.FLOAT64, nil, nil},
		//		//{"Far TrgOffWasOn", etensor.FLOAT64, nil, nil},
	}
	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			sch = append(sch, etable.Column{tn + " " + ts, etensor.FLOAT64, nil, nil})
		}
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", false, true, 0, true, 1)
	plt.SetColParams("PctCor", false, true, 0, true, 1)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			if ts == "Mem" {
				plt.SetColParams(tn+" "+ts, true, true, 0, true, 1) // default plot
			} else {
				plt.SetColParams(tn+" "+ts, false, true, 0, true, 1) // default plot
			}
		}
	}

	return plt
}

//////////////////////////////////////////////
//  TstCycLog

// LogTstCyc adds data from current trial to the TstCycLog table.
// log just has 100 cycles, is overwritten
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}

	dt.SetCellFloat("Cycle", cyc, float64(cyc))
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" Ge.Avg", cyc, float64(ly.Pools[0].Inhib.Ge.Avg))
		dt.SetCellFloat(ly.Nm+" Act.Avg", cyc, float64(ly.Pools[0].Inhib.Act.Avg))
	}

	if cyc%10 == 0 { // too slow to do every cyc
		// note: essential to use Go version of update when called from another goroutine
		ss.TstCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of activity etc over one trial by cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	np := 100 // max cycles
	sch := etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " Ge.Avg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + " Act.Avg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, np)
}

func (ss *Sim) ConfigTstCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", false, true, 0, false, 0)
	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" Ge.Avg", true, true, 0, true, .5)
		plt.SetColParams(lnm+" Act.Avg", true, true, 0, true, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epclog := ss.TstEpcLog
	epcix := etable.NewIdxView(epclog)
	// compute mean over last N epochs for run level
	nlast := 1
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast-1:]

	params := ss.RunName() // includes tag

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero))
	dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	dt.SetCellFloat("Close Mem", row, agg.Mean(epcix, "Close Mem")[0])
	dt.SetCellFloat("Close TrgOnWasOff", row, agg.Mean(epcix, "Close TrgOnWasOff")[0])
	dt.SetCellFloat("Close TrgOffWasOn", row, agg.Mean(epcix, "Close TrgOffWasOn")[0])

	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	split.Desc(spl, "Close Mem")
	split.Desc(spl, "Close TrgOnWasOff")
	split.Desc(spl, "Close TrgOffWasOn")
	ss.RunStats = spl.AggsToTable(false)

	// note: essential to use Go version of update when called from another goroutine
	ss.RunPlot.GoUpdate()
	if ss.RunFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.RunFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Close Mem", etensor.FLOAT64, nil, nil},
		{"Close TrgOnWasOff", etensor.FLOAT64, nil, nil},
		{"Close TrgOffWasOn", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("FirstZero", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", false, true, 0, true, 1)
	plt.SetColParams("PctCor", false, true, 0, true, 1)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	plt.SetColParams("Close Mem", true, true, 0, true, 1)         // default plot
	plt.SetColParams("Close TrgOnWasOff", true, true, 0, true, 1) // default plot
	plt.SetColParams("Close TrgOffWasOn", true, true, 0, true, 1) // default plot
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("hip")
	gi.SetAppAbout(`This demonstrates a basic Hippocampus model in Leabra. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewMainWindow("hip", "Hippocampus Close-Far", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = gi.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	// nv.Params.ColorMap = "Jet" // default is ColdHot
	// which fares pretty well in terms of discussion here:
	// https://matplotlib.org/tutorials/colors/colormaps.html
	nv.SetNet(ss.Net)
	ss.NetView = nv
	nv.ViewDefaults()

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnTrlPlot").(*eplot.Plot2D)
	ss.TrnTrlPlot = ss.ConfigTrnTrlPlot(plt, ss.TrnTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot").(*eplot.Plot2D)
	ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "AllOn", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.AllOn()
	})

	smen := gi.AddNewMenuButton(tbar, "load")
	smen.SetText("Load")

	smen.Menu.AddAction(gi.ActOpts{Label: "LoadSem", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.OpenPatsSem()
	})

	smen.Menu.AddAction(gi.ActOpts{Label: "LoadStudy", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.OpenPats()
	})

	smen.Menu.AddAction(gi.ActOpts{Label: "LoadRP", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.OpenPatsRP()
	})

	smen.Menu.AddAction(gi.ActOpts{Label: "Sleep", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.OpenPatsSleep()
	})

	trmen := gi.AddNewMenuButton(tbar, "train")
	trmen.SetText("Train")

	trmen.Menu.AddAction(gi.ActOpts{Label: "TrainHip", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.TrainHip()
		}
	})

	trmen.Menu.AddAction(gi.ActOpts{Label: "TrainSem", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.TrainSem()
		}
	})

	trmen.Menu.AddAction(gi.ActOpts{Label: "TrainRP", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.TrainRetrievalPractice()
		}
	})

	trmen.Menu.AddAction(gi.ActOpts{Label: "TrainSleep", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.TrainSleep()
		}
	})

	exmen := gi.AddNewMenuButton(tbar, "Exp")
	exmen.SetText("Exp")

	exmen.Menu.AddAction(gi.ActOpts{Label: "CloseUn", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.CloseUnrelated()
		}
	})

	exmen.Menu.AddAction(gi.ActOpts{Label: "CloseRe", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.CloseRelated()
		}
	})

	exmen.Menu.AddAction(gi.ActOpts{Label: "FarUn", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.FarUnrelated()
		}
	})

	exmen.Menu.AddAction(gi.ActOpts{Label: "FarRe", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.FarRelated()
		}
	})

	exmen.Menu.AddAction(gi.ActOpts{Label: "FullSet", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			ss.ViewOn = false
			ss.CloseUnrelated()
			ss.CloseRelated()
			ss.FarUnrelated()
			ss.FarRelated()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "StepHip", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrialHip()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "StepSem", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrialSem()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "TestTrialSem", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrialSem(false) // don't return on trial -- wrap
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "TestTrialH", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrialHip(false) // don't return on trial -- wrap
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Item", Icon: "step-fwd", Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		gi.StringPromptDialog(vp, "", "Test Item",
			gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				dlg := send.(*gi.Dialog)
				if sig == int64(gi.DialogAccepted) {
					val := gi.StringPromptDialogValue(dlg)
					idxs := ss.TestEnv.Table.RowsByString("Name", val, true, true) // contains, ignoreCase
					if len(idxs) == 0 {
						gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, true, false, nil, nil)
					} else {
						if !ss.IsRunning {
							ss.IsRunning = true
							fmt.Printf("testing index: %v\n", idxs[0])
							ss.TestItem(idxs[0])
							ss.IsRunning = false
							vp.SetNeedsFullRender()
						}
					}
				}
			})
	})

	tbar.AddAction(gi.ActOpts{Label: "TestTrialF", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrialFull(false) // don't return on trial -- wrap
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test AllSem", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAllSem()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test AllH", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAllHip()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test AllHPure", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAllHipPure()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test AllF", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAllFull()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Env", Icon: "gear", Tooltip: "select training input patterns: Close or Far."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			giv.CallMethod(ss, "SetEnv", vp)
		})

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "reset", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/leabra/blob/master/examples/ra25/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	inQuitPrompt := false
	gi.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, true, true,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// gi.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *gi.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, true, true,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
		{"SetEnv", ki.Props{
			"desc": "select which set of patterns to train on: Close or Far",
			"icon": "gear",
			"Args": ki.PropSlice{
				{"Train on Far", ki.Props{}},
			},
		}},
	},
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "aaaaa", "extra tag to add to file names saved from this run")
	flag.IntVar(&ss.MaxSubs, "subs", 1, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.IntVar(&ss.Group, "group", 0, "which set of subjects to run")
	flag.IntVar(&ss.SemWake, "sem", 6, "wake sem *100")
	flag.Float64Var(&ss.Outecback, "outec", 1, "out to ec prjn")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", false, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", false, "if true, save run epoch log to file")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.BoolVar(&ss.Daydream, "daydream", true, "whether to have day dream")
	flag.Parse()
	fmt.Printf("tag:" + ss.Tag + "\n")
	fmt.Printf("Running %d Subs %d Group %d sem %v outec\n", ss.MaxSubs, ss.Group, ss.SemWake, ss.Outecback)
	if ss.Daydream {
		fmt.Printf("daydream true")
	}
	ss.Init()

	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog {
		var err error
		fnm := ss.LogFileName("epc")
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving epoch log to: %v\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
	}
	if saveRunLog {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			fmt.Printf("Saving run log to: %v\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}

	ss.FullExp()

}

func mainrun() {
	TheSim.New()
	TheSim.Config()

	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		// gi.Update2DTrace = true
		TheSim.Init()
		win := TheSim.ConfigGui()
		win.StartEventLoop()
	}
}
