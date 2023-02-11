package activation

import (
	"main/internal/neural_net/domain/ports"
	"math"
)

// ReLU is a rectified linear unit activator
type ReLU struct{}

func NewReLUActivation() ports.Differentiable {
	return &ReLU{}
}

// F is ReLU(x)
func (a *ReLU) F(x float64) float64 { return math.Max(x, 0) }

// Df is ReLU'(y), where y = ReLU(x)
func (a *ReLU) Df(y float64) float64 {
	if y > 0 {
		return 1
	}
	return 0
}
