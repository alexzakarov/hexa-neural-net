package layer

import (
	"fmt"
	"main/internal/neural_net/application/services/layer/neuron"
	"main/internal/neural_net/application/services/layer/neuron/synapse"
	"main/internal/neural_net/domain/entities"
	"main/internal/neural_net/domain/ports"
	"main/internal/neural_net/domain/utils"
)

// Layer is a set of neurons and corresponding activation
type Layer struct {
	Neurons []ports.INeuron
	A       entities.ActivationType
}

// NewLayer creates a new layer with n nodes
func NewLayer(n int, activation entities.ActivationType) ports.ILayer {
	neurons := make([]ports.INeuron, n)

	for i := 0; i < n; i++ {
		act := activation
		if activation == entities.ActivationSoftmax {
			act = entities.ActivationLinear
		}
		neurons[i] = neuron.NewNeuron(act)
	}
	return &Layer{
		Neurons: neurons,
		A:       activation,
	}
}

func (l *Layer) GetNeurons() []ports.INeuron {
	return l.Neurons
}
func (l *Layer) GetA() entities.ActivationType {
	return l.A
}

func (l *Layer) SetNeurons(neurons []ports.INeuron) {
	l.Neurons = neurons
}

func (l *Layer) SetA(a entities.ActivationType) {
	l.A = a
}

func (l *Layer) AddNeuron(neuron ports.INeuron) {
	l.Neurons = append(l.Neurons, neuron)
}

func (l *Layer) Fire() {
	for _, n := range l.Neurons {
		n.Fire()
	}
	if l.A == entities.ActivationSoftmax {
		outs := make([]float64, len(l.Neurons))
		for i, _ := range l.Neurons {
			outs[i] = l.Neurons[i].GetValue()
		}
		sm := utils.Softmax(outs)
		for i, neuron := range l.Neurons {
			neuron.SetValue(sm[i])
		}
	}
}

// Connect fully connects layer l to next, and initializes each
// synapse with the given weight function
func (l *Layer) Connect(next ports.ILayer, weight synapse.WeightInitializer) {
	for i := range l.Neurons {
		for j := range next.GetNeurons() {
			syn := synapse.NewSynapse(weight())
			l.Neurons[i].AddOut(syn)
			next.GetNeurons()[j].AddIn(syn)
		}
	}
}

// ApplyBias creates and returns a bias synapse for each neuron in l
func (l *Layer) ApplyBias(weight synapse.WeightInitializer) []ports.ISynapse {
	biases := make([]ports.ISynapse, len(l.Neurons))
	for i := range l.Neurons {
		biases[i] = synapse.NewSynapse(weight())
		biases[i].SetIsBias(true)
		l.Neurons[i].AddIn(biases[i])
	}
	return biases
}

func (l *Layer) String() string {
	weights := make([][]float64, len(l.Neurons))
	for i, n := range l.Neurons {
		weights[i] = make([]float64, len(n.GetIn()))
		for j, s := range n.GetIn() {
			weights[i][j] = s.GetWeight()
		}
	}
	return fmt.Sprintf("%+v", weights)
}
