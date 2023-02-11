package synapse

// Synapse is an edge between neurons
type Synapse struct {
	Weight  float64
	In, Out float64 `json:"-"`
	IsBias  bool
}

// NewSynapse returns a synapse with the specified initialized weight
func NewSynapse(weight float64) *Synapse {
	return &Synapse{Weight: weight}
}

func (s *Synapse) GetWeight() float64 {
	return s.Weight
}

func (s *Synapse) GetIn() float64 {
	return s.In
}

func (s *Synapse) GetOut() float64 {
	return s.Out
}

func (s *Synapse) GetIsBias() bool {
	return s.IsBias
}

func (s *Synapse) SetWeight(weight float64) {
	s.Weight = weight
}

func (s *Synapse) SetIn(in float64) {
	s.In = in
}

func (s *Synapse) SetOut(out float64) {
	s.Out = out
}

func (s *Synapse) SetIsBias(isBias bool) {
	s.IsBias = isBias
}

func (s *Synapse) Fire(value float64) {
	s.In = value
	s.Out = s.In * s.Weight
}
