package bayesopt

import (
	"math/rand"
	"sync"

	"github.com/c-bata/goptuna"
	bo "github.com/d4l3k/go-bayesopt"
)

// Sampler returns the next search points
func NewSampler(opts ...SamplerOption) *Sampler {
	s := &Sampler{
		rng: rand.New(rand.NewSource(0)),
	}

	for _, opt := range opts {
		opt(s)
	}
	return s
}

// SamplerOption is a type of function to set change the option.
type SamplerOption func(sampler *Sampler)

// SamplerOptionSeed sets seed number.
func SamplerOptionSeed(seed int64) SamplerOption {
	return func(sampler *Sampler) {
		sampler.rng = rand.New(rand.NewSource(seed))
	}
}

// SamplerOptionOptimizerOptions sets options for d4l3k/go-bayesopt's optimizer.
func SamplerOptionOptimizerOptions(opts []bo.OptimizerOption) SamplerOption {
	return func(sampler *Sampler) {
		sampler.opts = opts
	}
}

// Sampler for Gaussian Process based Bayesian Optimization.
type Sampler struct {
	opts []bo.OptimizerOption
	rng  *rand.Rand
	mu   sync.Mutex
}

var _ goptuna.RelativeSampler = &Sampler{}

// SampleRelative samples multiple dimensional parameters in a given search space.
func (s *Sampler) SampleRelative(
	study *goptuna.Study,
	trial goptuna.FrozenTrial,
	searchSpace map[string]interface{},
) (map[string]float64, error) {
	params := make([]bo.Param, 0, len(searchSpace))
	for name := range searchSpace {
		distribution := searchSpace[name]
		switch d := distribution.(type) {
		case goptuna.UniformDistribution:
			params = append(params, bo.UniformParam{
				Name: name,
				Max:  d.High,
				Min:  d.Low,
			})
			continue
		}
	}

	optimizer := bo.New(params, s.opts...)
	trials, err := study.GetTrials()
	if err != nil {
		return nil, err
	}

	for i := range trials {
		if trial.State != goptuna.TrialStateComplete {
			continue
		}

		x := generateParam(params, trials[i])
		if x == nil {
			continue
		}

		optimizer.Log(x, trial.Value)
	}

	boNextParam, _, err := optimizer.Next()
	if err != nil {
		return nil, err
	}

	sampled := make(map[string]float64, len(boNextParam))
	for p := range boNextParam {
		sampled[p.GetName()] = boNextParam[p]
	}
	return sampled, nil
}

func generateParam(params []bo.Param, trial goptuna.FrozenTrial) map[bo.Param]float64 {
	x := make(map[bo.Param]float64, len(params))

	for _, p := range params {
		distributionInterface, ok := trial.Distributions[p.GetName()]
		if !ok {
			return nil
		}

		xr, ok := trial.Params[p.GetName()]
		if !ok {
			return nil
		}

		// This function might be removed at https://github.com/c-bata/goptuna/pull/78/
		ir, err := goptuna.ToInternalRepresentation(distributionInterface, xr)
		if err != nil {
			return nil
		}

		switch distributionInterface.(type) {
		case goptuna.UniformDistribution:
			var distribution goptuna.UniformDistribution
			distribution, ok = distributionInterface.(goptuna.UniformDistribution)
			if !ok {
				return nil
			}
			if !distribution.Contains(ir) {
				return nil
			}
		default:
			// Currently this sampler supports UniformDistribution only.
			return nil
		}
		x[p] = ir
	}
	return x
}
