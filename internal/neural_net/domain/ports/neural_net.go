package ports

import (
	"main/internal/neural_net/application/services/layer/neuron/synapse"
	"main/internal/neural_net/domain/entities"
)

type ILayer interface {
	Fire()
	Connect(ILayer, synapse.WeightInitializer)
	ApplyBias(synapse.WeightInitializer) []ISynapse
	String() string
	GetNeurons() []INeuron
	GetA() entities.ActivationType
	SetNeurons([]INeuron)
	SetA(entities.ActivationType)
	AddNeuron(INeuron)
}

// Differentiable is an activation function and its first order derivative,
// where the latter is expressed as a function of the former for efficiency
type Differentiable interface {
	F(float64) float64
	Df(float64) float64
}

type ISynapse interface {
	Fire(float64)
	GetWeight() float64
	GetIn() float64
	GetOut() float64
	GetIsBias() bool
	SetWeight(float64)
	SetIn(float64)
	SetOut(float64)
	SetIsBias(bool)
}

type INeuron interface {
	Fire()
	Activate(float64) float64
	DActivate(float64) float64
	GetA() entities.ActivationType
	GetIn() []ISynapse
	GetOut() []ISynapse
	GetValue() float64
	SetA(entities.ActivationType)
	SetIn([]ISynapse)
	SetOut([]ISynapse)
	SetValue(float64)
	AddIn(ISynapse)
	AddOut(ISynapse)
}

type INeuralNet interface {
	Fire()
	Forward([]float64) error
	Predict([]float64) []float64
	NumWeights() int
	String() string
	ApplyWeights([][][]float64)
	Weights() [][][]float64
	Dump() *entities.Dump
	GetLayers() []ILayer
	GetBiases() [][]ISynapse
	GetConfig() entities.Config
}
