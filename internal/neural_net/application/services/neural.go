package services

import (
	"fmt"
	"main/internal/neural_net/application/services/layer"
	"main/internal/neural_net/application/services/layer/neuron/synapse"
	"main/internal/neural_net/domain/entities"
	"main/internal/neural_net/domain/ports"
	"main/internal/neural_net/domain/utils"
)

// Neural is a neural network
type Neural struct {
	Layers []ports.ILayer
	Biases [][]ports.ISynapse
	Config entities.Config
}

// NewNeural returns a new neural network
func NewNeural(c entities.Config) ports.INeuralNet {

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

	layers := InitializeLayers(c)

	var biases [][]ports.ISynapse
	if c.Bias {
		biases = make([][]ports.ISynapse, len(layers))
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

func (n *Neural) GetLayers() []ports.ILayer {
	return n.Layers
}

func (n *Neural) GetBiases() [][]ports.ISynapse {
	return n.Biases
}

func (n *Neural) GetConfig() entities.Config {
	return n.Config
}
func InitializeLayers(c entities.Config) []ports.ILayer {
	layers := make([]ports.ILayer, len(c.Layout))
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

	for _, neuron := range layers[0].GetNeurons() {
		neuron.SetIn(make([]ports.ISynapse, c.Inputs))
		for i := range neuron.GetIn() {
			neuron.GetIn()[i] = synapse.NewSynapse(c.Weight())
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
	for _, n := range n.Layers[0].GetNeurons() {
		for i := 0; i < len(input); i++ {
			n.GetIn()[i].Fire(input[i])
		}
	}
	n.Fire()
	return nil
}

// Predict computes a forward pass and returns a prediction
func (n *Neural) Predict(input []float64) []float64 {
	n.Forward(input)

	outLayer := n.Layers[len(n.Layers)-1]
	out := make([]float64, len(outLayer.GetNeurons()))
	for i, neuron := range outLayer.GetNeurons() {
		out[i] = neuron.GetValue()
	}
	return out
}

// NumWeights returns the number of weights in the network
func (n *Neural) NumWeights() (num int) {
	for _, l := range n.Layers {
		for _, n := range l.GetNeurons() {
			num += len(n.GetIn())
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

// ApplyWeights sets the weights from a three-dimensional slice
func (n *Neural) ApplyWeights(weights [][][]float64) {
	for i, l := range n.Layers {
		for j := range l.GetNeurons() {
			for k := range l.GetNeurons()[j].GetIn() {
				n.Layers[i].GetNeurons()[j].GetIn()[k].SetWeight(weights[i][j][k])
			}
		}
	}
}

// Weights returns all weights in sequence
func (n *Neural) Weights() [][][]float64 {
	weights := make([][][]float64, len(n.Layers))
	for i, l := range n.Layers {
		weights[i] = make([][]float64, len(l.GetNeurons()))
		for j, n := range l.GetNeurons() {
			weights[i][j] = make([]float64, len(n.GetIn()))
			for k, in := range n.GetIn() {
				weights[i][j][k] = in.GetWeight()
			}
		}
	}
	return weights
}

// Dump generates a network dump
func (n *Neural) Dump() *entities.Dump {
	return &entities.Dump{
		Config:  n.Config,
		Weights: n.Weights(),
	}
}
