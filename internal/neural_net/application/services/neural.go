package services

import (
	"fmt"
	"main/internal/neural_net/application/services/layer"
	"main/internal/neural_net/application/services/layer/neuron/synapse"
	"main/internal/neural_net/domain/entities"
	"main/internal/neural_net/domain/utils"
)

// Neural is a neural network
type Neural struct {
	Layers []*layer.Layer
	Biases [][]*synapse.Synapse
	Config *entities.Config
}

// NewNeural returns a new neural network
func NewNeural(c *entities.Config) *Neural {

	if c.Weight == nil {
		c.Weight = synapse.NewUniform(0.5, 0)
	}
	if c.Activation == entities.ActivationNone {
		c.Activation = entities.ActivationSigmoid
	}
	if c.Loss == entities.LossNone {
		switch c.Mode {
		case entities.ModeMultiClass, entities.ModeMultiLabel:
			c.Loss = entities.LossCrossEntropy
		case entities.ModeBinary:
			c.Loss = entities.LossBinaryCrossEntropy
		default:
			c.Loss = entities.LossMeanSquared
		}
	}

	layers := initializeLayers(c)

	var biases [][]*synapse.Synapse
	if c.Bias {
		biases = make([][]*synapse.Synapse, len(layers))
		for i := 0; i < len(layers); i++ {
			if c.Mode == entities.ModeRegression && i == len(layers)-1 {
				continue
			}
			biases[i] = layers[i].ApplyBias(c.Weight)
		}
	}

	return &Neural{
		Layers: layers,
		Biases: biases,
		Config: c,
	}
}

func initializeLayers(c *entities.Config) []*layer.Layer {
	layers := make([]*layer.Layer, len(c.Layout))
	for i := range layers {
		act := c.Activation
		if i == (len(layers)-1) && c.Mode != entities.ModeDefault {
			act = utils.OutputActivation(c.Mode)
		}
		layers[i] = layer.NewLayer(c.Layout[i], act)
	}

	for i := 0; i < len(layers)-1; i++ {
		layers[i].Connect(layers[i+1], c.Weight)
	}

	for _, neuron := range layers[0].Neurons {
		neuron.In = make([]*synapse.Synapse, c.Inputs)
		for i := range neuron.In {
			neuron.In[i] = synapse.NewSynapse(c.Weight())
		}
	}

	return layers
}

func (n *Neural) Fire() {
	for _, b := range n.Biases {
		for _, s := range b {
			s.Fire(1)
		}
	}
	for _, l := range n.Layers {
		l.Fire()
	}
}

// Forward computes a forward pass
func (n *Neural) Forward(input []float64) error {
	if len(input) != n.Config.Inputs {
		return fmt.Errorf("Invalid input dimension - expected: %d got: %d", n.Config.Inputs, len(input))
	}
	for _, n := range n.Layers[0].Neurons {
		for i := 0; i < len(input); i++ {
			n.In[i].Fire(input[i])
		}
	}
	n.Fire()
	return nil
}

// Predict computes a forward pass and returns a prediction
func (n *Neural) Predict(input []float64) []float64 {
	n.Forward(input)

	outLayer := n.Layers[len(n.Layers)-1]
	out := make([]float64, len(outLayer.Neurons))
	for i, neuron := range outLayer.Neurons {
		out[i] = neuron.Value
	}
	return out
}

// NumWeights returns the number of weights in the network
func (n *Neural) NumWeights() (num int) {
	for _, l := range n.Layers {
		for _, n := range l.Neurons {
			num += len(n.In)
		}
	}
	return
}

func (n *Neural) String() string {
	var s string
	for _, l := range n.Layers {
		s = fmt.Sprintf("%s\n%s", s, l)
	}
	return s
}
