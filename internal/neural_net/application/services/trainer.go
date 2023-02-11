package services

import (
	"main/internal/neural_net/application/services/solver"
	"main/internal/neural_net/domain/ports"
	"time"
)

// Trainer is a neural network trainer
type Trainer interface {
	Train(n *Neural, examples, validation Examples, iterations int)
}

// OnlineTrainer is a basic, online network trainer
type OnlineTrainer struct {
	*internal
	neuralNet ports.INeuralNet
	solver    solver.Solver
	printer   *StatsPrinter
	verbosity int
}

// NewTrainer creates a new trainer
func NewTrainer(n ports.INeuralNet, solver solver.Solver, verbosity int) *OnlineTrainer {
	return &OnlineTrainer{
		neuralNet: n,
		solver:    solver,
		printer:   NewStatsPrinter(),
		verbosity: verbosity,
	}
}

type internal struct {
	deltas [][]float64
}

func newTraining(layers []ports.ILayer) *internal {
	deltas := make([][]float64, len(layers))
	for i, l := range layers {
		deltas[i] = make([]float64, len(l.GetNeurons()))
	}
	return &internal{
		deltas: deltas,
	}
}

// Train trains n
func (t *OnlineTrainer) Train(examples, validation Examples, iterations int) {
	t.internal = newTraining(t.neuralNet.GetLayers())

	train := make(Examples, len(examples))
	copy(train, examples)

	t.printer.Init(t.neuralNet)
	t.solver.Init(t.neuralNet.NumWeights())

	ts := time.Now()
	for i := 1; i <= iterations; i++ {
		examples.Shuffle()
		for j := 0; j < len(examples); j++ {
			t.learn(examples[j], i)
		}
		if t.verbosity > 0 && i%t.verbosity == 0 && len(validation) > 0 {
			t.printer.PrintProgress(t.neuralNet, validation, time.Since(ts), i)
		}
	}
}

func (t *OnlineTrainer) learn(e Example, it int) {
	t.neuralNet.Forward(e.Input)
	t.calculateDeltas(e.Response)
	t.update(it)
}

func (t *OnlineTrainer) calculateDeltas(ideal []float64) {
	for i, neuron := range t.neuralNet.GetLayers()[len(t.neuralNet.GetLayers())-1].GetNeurons() {
		t.deltas[len(t.neuralNet.GetLayers())-1][i] = GetLoss(t.neuralNet.GetConfig().Loss).Df(
			neuron.GetValue(),
			ideal[i],
			neuron.DActivate(neuron.GetValue()))
	}

	for i := len(t.neuralNet.GetLayers()) - 2; i >= 0; i-- {
		for j, neuron := range t.neuralNet.GetLayers()[i].GetNeurons() {
			var sum float64
			for k, s := range neuron.GetOut() {
				sum += s.GetWeight() * t.deltas[i+1][k]
			}
			t.deltas[i][j] = neuron.DActivate(neuron.GetValue()) * sum
		}
	}
}

func (t *OnlineTrainer) update(it int) {
	var idx int
	for i, l := range t.neuralNet.GetLayers() {
		for j := range l.GetNeurons() {
			for k := range l.GetNeurons()[j].GetIn() {
				update := t.solver.Update(l.GetNeurons()[j].GetIn()[k].GetWeight(),
					t.deltas[i][j]*l.GetNeurons()[j].GetIn()[k].GetIn(),
					it,
					idx)
				l.GetNeurons()[j].GetIn()[k].SetWeight(
					l.GetNeurons()[j].GetIn()[k].GetWeight() +
						update)
				idx++
			}
		}
	}
}
