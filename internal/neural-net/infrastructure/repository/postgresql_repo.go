package repository

import (
	"github.com/jackc/pgx/v4/pgxpool"
	"main/internal/neural-net/domain/ports"
)

// postgresqlRepo Struct
type postgresqlRepo struct {
	db *pgxpool.Pool
}

// NewPostgresqlRepository Auth Domain postgresql repository constructor
func NewPostgresqlRepository(db *pgxpool.Pool) ports.IPostgresqlRepository {
	return &postgresqlRepo{db: db}
}
