package services

import (
	"encoding/json"
	"main/internal/neural_net/domain/entities"
	"main/internal/neural_net/domain/ports"
)

// FromDump restores a Neural from a dump
func FromDump(dump *entities.Dump) ports.INeuralNet {
	n := NewNeural(dump.Config)
	n.ApplyWeights(dump.Weights)

	return n
}

// Marshal marshals to JSON from network
func (n Neural) Marshal() ([]byte, error) {
	return json.Marshal(n.Dump())
}

// Unmarshal restores network from a JSON blob
func Unmarshal(bytes []byte) (ports.INeuralNet, error) {
	var dump entities.Dump
	if err := json.Unmarshal(bytes, &dump); err != nil {
		return nil, err
	}
	return FromDump(&dump), nil
}
