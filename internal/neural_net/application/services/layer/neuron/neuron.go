package neuron

import (
	"main/internal/neural_net/domain/entities"
	"main/internal/neural_net/domain/ports"
	"main/internal/neural_net/domain/utils"
)

// Neuron is a neural network node
type Neuron struct {
	A     entities.ActivationType `json:"-"`
	In    []ports.ISynapse
	Out   []ports.ISynapse
	Value float64 `json:"-"`
}

// NewNeuron returns a neuron with the given activation
func NewNeuron(activation entities.ActivationType) ports.INeuron {
	return &Neuron{
		A:     activation,
		In:    []ports.ISynapse{},
		Out:   []ports.ISynapse{},
		Value: 0,
	}
}

func (n *Neuron) GetA() entities.ActivationType {
	return n.A
}

func (n *Neuron) GetIn() []ports.ISynapse {
	return n.In
}

func (n *Neuron) GetOut() []ports.ISynapse {
	return n.Out
}

func (n *Neuron) GetValue() float64 {
	return n.Value
}

func (n *Neuron) SetA(a entities.ActivationType) {
	n.A = a
}

func (n *Neuron) SetIn(synapse []ports.ISynapse) {
	n.In = synapse
}

func (n *Neuron) SetOut(synapse []ports.ISynapse) {
	n.Out = synapse
}

func (n *Neuron) SetValue(value float64) {
	n.Value = value
}

func (n *Neuron) AddIn(synapse ports.ISynapse) {
	n.In = append(n.In, synapse)
}

func (n *Neuron) AddOut(synapse ports.ISynapse) {
	n.Out = append(n.Out, synapse)
}

func (n *Neuron) Fire() {
	var sum float64
	for _, s := range n.In {
		sum += s.GetOut()
	}
	n.Value = n.Activate(sum)

	nVal := n.Value
	for _, s := range n.Out {
		s.Fire(nVal)
	}
}

// Activate applies the neurons activation
func (n *Neuron) Activate(x float64) float64 {
	return utils.GetActivation(n.A).F(x)
}

// DActivate applies the derivative of the neurons activation
func (n *Neuron) DActivate(x float64) float64 {
	return utils.GetActivation(n.A).Df(x)
}
