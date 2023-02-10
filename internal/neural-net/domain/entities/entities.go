package entities

type GetReq struct {
	Src         int8   `json:"src" validate:"required,min=1,max=5"`
	UserType    int8   `json:"user_type" validate:"required,min=1,max=5"`
	UserTitle   string `json:"user_title" validate:"required"`
	CompanyName string `json:"company_name,omitempty"`
	UserName    string `json:"user_name" validate:"required,min=10,max=50"`
	UserPass    string `json:"user_pass,omitempty"`
	UserPhone   string `json:"user_phone" validate:"required,min=10,max=20"`
	VerifyCode  int64  `json:"verify_code,omitempty"`
}

type HandlerResponse struct {
	Error   bool        `json:"error"`
	Message string      `json:"message"`
	Data    interface{} `json:"data"`
}
