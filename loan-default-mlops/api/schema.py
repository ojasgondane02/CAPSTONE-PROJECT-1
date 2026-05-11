from pydantic import BaseModel

class LoanApplication(BaseModel):
    # Defining input features expected by the model
    ID: int
    year: int
    loan_limit: str
    Gender: str
    approv_in_adv: str
    loan_type: str
    loan_purpose: str
    Credit_Worthiness: str
    open_credit: str
    business_or_commercial: str
    loan_amount: int
    rate_of_interest: float
    Interest_rate_spread: float
    Upfront_charges: float
    term: float
    Neg_ammortization: str
    interest_only: str
    lump_sum_payment: str
    property_value: float
    construction_type: str
    occupancy_type: str
    Secured_by: str
    total_units: str
    income: float
    credit_type: str
    Credit_Score: int
    co_applicant_credit_type: str
    age: str
    submission_of_application: str
    LTV: float
    Region: str
    Security_Type: str
    dtir1: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "ID": 24890,
                "year": 2019,
                "loan_limit": "cf",
                "Gender": "Sex Not Available",
                "approv_in_adv": "nopre",
                "loan_type": "type1",
                "loan_purpose": "p1",
                "Credit_Worthiness": "l1",
                "open_credit": "nopc",
                "business_or_commercial": "nob/c",
                "loan_amount": 116500,
                "rate_of_interest": 3.99,
                "Interest_rate_spread": 0.2,
                "Upfront_charges": 0.0,
                "term": 360.0,
                "Neg_ammortization": "not_neg",
                "interest_only": "not_int",
                "lump_sum_payment": "not_lpsm",
                "property_value": 118000.0,
                "construction_type": "sb",
                "occupancy_type": "pr",
                "Secured_by": "home",
                "total_units": "1U",
                "income": 1740.0,
                "credit_type": "EXP",
                "Credit_Score": 758,
                "co_applicant_credit_type": "CIB",
                "age": "25-34",
                "submission_of_application": "to_inst",
                "LTV": 98.72,
                "Region": "south",
                "Security_Type": "direct",
                "dtir1": 45.0
            }
        }
