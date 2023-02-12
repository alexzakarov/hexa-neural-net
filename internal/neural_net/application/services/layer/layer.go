package layer

import (
	"fmt"
	"main/internal/neural_net/application/services/layer/neuron"
	"main/internal/neural_net/application/services/layer/neuron/synapse"
	"main/internal/neural_net/domain/entities"
	"main/internal/neural_net/domain/utils"
)

// Layer is a set of neurons and corresponding activation
type Layer struct {
	Neurons []*neuron.Neuron
	A       entities.ActivationType
}

// NewLayer creates a new layer with n nodes
func NewLayer(n int, activation entities.ActivationType) *Layer {
	neurons := make([]*neuron.Neuron, n)

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

func (l *Layer) Fire() {
	for _, n := range l.Neurons {
		n.Fire()
	}
	if l.A == entities.ActivationSoftmax {
		outs := make([]float64, len(l.Neurons))
		for i, neuron := range l.Neurons {
			outs[i] = neuron.Value
		}
		sm := utils.Softmax(outs)
		for i, neuron := range l.Neurons {
			neuron.Value = sm[i]
		}
	}
}

// Connect fully connects layer l to next, and initializes each
// synapse with the given weight function
func (l *Layer) Connect(next *Layer, weight synapse.WeightInitializer) {
	for i := range l.Neurons {
		for j := range next.Neurons {
			syn := synapse.NewSynapse(weight())
			l.Neurons[i].Out = append(l.Neurons[i].Out, syn)
			next.Neurons[j].In = append(next.Neurons[j].In, syn)
		}
	}
}

// ApplyBias creates and returns a bias synapse for each neuron in l
func (l *Layer) ApplyBias(weight synapse.WeightInitializer) []*synapse.Synapse {
	biases := make([]*synapse.Synapse, len(l.Neurons))
	for i := range l.Neurons {
		biases[i] = synapse.NewSynapse(weight())
		biases[i].IsBias = true
		l.Neurons[i].In = append(l.Neurons[i].In, biases[i])
	}
	return biases
}

func (l Layer) String() string {
	weights := make([][]float64, len(l.Neurons))
	for i, n := range l.Neurons {
		weights[i] = make([]float64, len(n.In))
		for j, s := range n.In {
			weights[i][j] = s.Weight
		}
	}
	return fmt.Sprintf("%+v", weights)
}
