"""AI contract intelligence analyzer."""
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date
import json

class ContractObligation(BaseModel):
    party: str
    obligation: str
    deadline: Optional[str] = None
    penalty: Optional[str] = None
    risk_level: str = "low"  # low, medium, high

class ContractAnalysis(BaseModel):
    contract_type: str
    parties: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    total_value: Optional[str] = None
    obligations: List[ContractObligation]
    risk_score: float = Field(ge=0, le=10)
    risk_factors: List[str]
    lgpd_compliant: bool
    recommendations: List[str]

BR_CONTRACT_SYSTEM = """Voce e um especialista em direito contratual brasileiro.
Analise contratos considerando: CLT, LGPD, Codigo Civil, CDC.
Identifique riscos, obrigacoes e pontos de negociacao.
Sempre retorne JSON valido."""

class ContractAnalyzer:
    def __init__(self):
        self.llm = ChatVertexAI(model_name="gemini-1.5-pro-002", temperature=0)

    def analyze(self, contract_text: str) -> ContractAnalysis:
        prompt = f"""{BR_CONTRACT_SYSTEM}

Contrato:
{contract_text[:6000]}

Retorne JSON com: contract_type, parties, start_date, end_date, total_value,
obligations (list com: party, obligation, deadline, penalty, risk_level),
risk_score (0-10), risk_factors (list), lgpd_compliant (bool), recommendations (list)"""
        resp = self.llm.invoke(prompt).content
        data = json.loads(resp.split("```json")[-1].split("```")[0] if "```" in resp else resp)
        return ContractAnalysis(**data)

    def compare_clauses(self, contract_text: str, clause_type: str) -> dict:
        """Compare specific clause against market standards."""
        prompt = f"""Compare esta clausula de {clause_type} do contrato com os padroes do mercado brasileiro.
Contrato: {contract_text[:2000]}
Retorne: {{"clause_found": str, "market_standard": str, "deviation_risk": str, "recommendation": str}}"""
        resp = self.llm.invoke(prompt).content
        return json.loads(resp.split("```json")[-1].split("```")[0] if "```" in resp else resp)
