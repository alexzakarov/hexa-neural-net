package http

import (
	"github.com/gofiber/fiber/v2"
	"main/internal/auth/domain/ports"
)

// MapRoutes Auth Domain routes
func MapRoutes(h ports.IHandlers, router fiber.Router) {
	//neuralNet := router.Group("/auth")

	/* Example HTTP handler Methods */
	//auth.Get("/:id", h.Login)
	//auth.Delete("/:id", h.Login)
	//auth.Put("/:id", h.Login)

}
