package activation

import (
	"main/internal/neural_net/domain/ports"
)

// Linear is a linear activator
type Linear struct{}

func NewLinearActivation() ports.Differentiable {
	return &Linear{}
}

// F is the identity function
func (a *Linear) F(x float64) float64 { return x }

// Df is constant
func (a *Linear) Df(x float64) float64 { return 1 }
