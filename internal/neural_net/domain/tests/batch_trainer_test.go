package tests

import (
	"main/internal/neural_net/application/services"
	"main/internal/neural_net/application/services/layer/neuron/synapse"
	"main/internal/neural_net/application/services/solver"
	"main/internal/neural_net/domain/entities"
	"math/rand"
	"runtime"
	"testing"
	"time"
)

func Benchmark_xor(b *testing.B) {
	rand.Seed(time.Now().Unix())
	n := services.NewNeural(entities.Config{
		Inputs:     2,
		Layout:     []int{3, 3, 1},
		Activation: entities.ActivationSigmoid,
		Mode:       entities.ModeBinary,
		Weight:     synapse.NewUniform(.25, 0),
		Bias:       true,
	})
	exs := services.Examples{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
	}
	const minExamples = 4000
	var dupExs services.Examples
	for len(dupExs) < minExamples {
		dupExs = append(dupExs, exs...)
	}

	for i := 0; i < b.N; i++ {
		const iterations = 20
		solver := solver.NewAdam(0.001, 0.9, 0.999, 1e-100)
		trainer := services.NewBatchTrainer(n, solver, iterations, len(dupExs)/2, runtime.NumCPU())
		trainer.Train(dupExs, dupExs, iterations)
	}
}
