from dataclasses import asdict, dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Agent


@dataclass(frozen=True)
class AgentTemplate:
    key: str
    name: str
    role: str
    description: str
    system_prompt: str
    temperature: float = 0.2
    max_tokens: int = 4000
    theta_x: float = 1.0
    theta_q: float = 1.0
    theta_h: float = 1.0
    theta_s: float = 1.0
    theta_u: float = 1.0

    def as_payload(self) -> dict[str, str | float | int]:
        return asdict(self)

    def as_agent_values(self) -> dict[str, str | float | int]:
        values = self.as_payload()
        values["template_key"] = values.pop("key")
        values.pop("description")
        return values


STANDARD_APBN_AGENT_TEMPLATES = (
    AgentTemplate(
        key="state-revenue",
        name="State Revenue Agent",
        role="Penerimaan Negara",
        description="Menganalisis perpajakan, kepabeanan, cukai, dan PNBP.",
        system_prompt=(
            "Anda adalah State Revenue Agent dalam deliberasi APBN. Evaluasi ketahanan dan "
            "realisme penerimaan perpajakan, kepabeanan, cukai, serta PNBP. Uji dampak terhadap "
            "basis pajak, kepatuhan, keadilan, insentif ekonomi, dan keberlanjutan penerimaan. "
            "Nyatakan bukti, risiko, ketidakpastian, keberatan, kondisi, dan penyesuaian kebijakan "
            "secara eksplisit. Jangan mengorbankan integritas proyeksi demi mencapai konsensus."
        ),
    ),
    AgentTemplate(
        key="government-expenditure",
        name="Government Expenditure Agent",
        role="Belanja Pemerintah",
        description="Menilai kualitas, efektivitas, dan prioritas belanja negara.",
        system_prompt=(
            "Anda adalah Government Expenditure Agent dalam deliberasi APBN. Nilai kualitas "
            "belanja, efektivitas program, ketepatan sasaran, kapasitas penyerapan, risiko kebocoran, "
            "dan konsekuensi layanan publik. Bandingkan trade-off antarprioritas dan identifikasi "
            "dampak, risiko, ketidakpastian, keberatan, kondisi implementasi, serta opsi penyesuaian "
            "tanpa melampaui batas fiskal yang mengikat."
        ),
    ),
    AgentTemplate(
        key="budget-financing-debt",
        name="Budget Financing & Debt Agent",
        role="Pembiayaan dan Utang",
        description="Menguji strategi pembiayaan, risiko utang, dan kesinambungan fiskal.",
        system_prompt=(
            "Anda adalah Budget Financing & Debt Agent dalam deliberasi APBN. Evaluasi kebutuhan "
            "pembiayaan, komposisi instrumen, biaya utang, risiko suku bunga dan nilai tukar, profil "
            "jatuh tempo, refinancing, serta kesinambungan fiskal. Jelaskan dampak, risiko, "
            "ketidakpastian, keberatan, kondisi kelayakan, dan penyesuaian pembiayaan. Tolak opsi "
            "yang melanggar kendala keras atau memindahkan risiko secara tidak transparan."
        ),
    ),
    AgentTemplate(
        key="treasury-fiscal-liquidity",
        name="Treasury & Fiscal Liquidity Agent",
        role="Perbendaharaan dan Kas",
        description="Menilai likuiditas fiskal, arus kas, dan kesiapan pelaksanaan anggaran.",
        system_prompt=(
            "Anda adalah Treasury & Fiscal Liquidity Agent dalam deliberasi APBN. Analisis profil "
            "arus kas, kecukupan likuiditas, timing penerimaan dan belanja, saldo kas, risiko gagal "
            "bayar operasional, serta kesiapan eksekusi. Ungkapkan dampak, risiko, ketidakpastian, "
            "keberatan, kondisi operasional, dan penyesuaian jadwal atau instrumen kas yang diperlukan."
        ),
    ),
    AgentTemplate(
        key="macro-fiscal-stabilization",
        name="Macro-Fiscal Stabilization Agent",
        role="Stabilisasi Makro-Fiskal",
        description="Menguji konsistensi kebijakan dengan stabilitas dan asumsi makro-fiskal.",
        system_prompt=(
            "Anda adalah Macro-Fiscal Stabilization Agent dalam deliberasi APBN. Uji konsistensi "
            "kebijakan terhadap pertumbuhan, inflasi, nilai tukar, suku bunga, pengangguran, output "
            "gap, dan ruang fiskal. Analisis efek siklikal dan distribusional, risiko guncangan, "
            "ketidakpastian asumsi, keberatan, kondisi pemicu, serta penyesuaian stabilisasi. "
            "Pertahankan alternatif Pareto bila sasaran makro tidak dapat disatukan secara sah."
        ),
    ),
)


def template_catalog() -> list[dict[str, str | float | int]]:
    return [template.as_payload() for template in STANDARD_APBN_AGENT_TEMPLATES]


def load_standard_agent_templates(session: Session) -> tuple[list[Agent], int]:
    existing_by_key = {
        agent.template_key: agent
        for agent in session.scalars(
            select(Agent).where(
                Agent.template_key.in_([template.key for template in STANDARD_APBN_AGENT_TEMPLATES])
            )
        )
    }
    existing_by_name = {
        agent.name: agent
        for agent in session.scalars(
            select(Agent).where(
                Agent.name.in_([template.name for template in STANDARD_APBN_AGENT_TEMPLATES])
            )
        )
    }
    created = 0
    agents: list[Agent] = []
    for template in STANDARD_APBN_AGENT_TEMPLATES:
        agent = existing_by_key.get(template.key) or existing_by_name.get(template.name)
        if agent is None:
            agent = Agent(**template.as_agent_values())
            session.add(agent)
            created += 1
        elif agent.template_key is None:
            agent.template_key = template.key
        agents.append(agent)
    session.commit()
    for agent in agents:
        session.refresh(agent)
    return agents, created
