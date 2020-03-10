package main

import (
	"log"
	"math"

	"github.com/c-bata/goptuna"
	bayesopt "github.com/c-bata/goptuna-bayesopt"
)

func objective(trial goptuna.Trial) (float64, error) {
	x1, _ := trial.SuggestUniform("x1", -10, 10)
	x2, _ := trial.SuggestUniform("x2", -10, 10)
	return math.Pow(x1-2, 2) + math.Pow(x2+5, 2), nil
}

func main() {
	relativeSampler := bayesopt.NewSampler()
	study, err := goptuna.CreateStudy(
		"goptuna-example",
		goptuna.StudyOptionRelativeSampler(relativeSampler),
	)
	if err != nil {
		log.Fatal("failed to create study:", err)
	}

	if err = study.Optimize(objective, 50); err != nil {
		log.Fatal("failed to optimize:", err)
	}

	v, _ := study.GetBestValue()
	params, _ := study.GetBestParams()
	log.Printf("Best evaluation=%f (x1=%f, x2=%f)",
		v, params["x1"].(float64), params["x2"].(float64))
}
