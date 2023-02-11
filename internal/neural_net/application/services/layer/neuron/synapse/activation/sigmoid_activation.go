package activation

import (
	"main/internal/neural_net/domain/ports"
	"math"
)

// Sigmoid is a logistic activator in the special case of a = 1
type Sigmoid struct{}

func NewSigmoidActivation() ports.Differentiable {
	return &Sigmoid{}
}

// F is Sigmoid(x)
func (a *Sigmoid) F(x float64) float64 { return Logistic(x, 1) }

// Df is Sigmoid'(y), where y = Sigmoid(x)
func (a *Sigmoid) Df(y float64) float64 { return y * (1 - y) }

// Logistic is the logistic function
func Logistic(x, a float64) float64 {
	return 1 / (1 + math.Exp(-a*x))
}
