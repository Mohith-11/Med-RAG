# -*- coding: utf-8 -*-
"""
evaluate_200q.py  --  Advanced RAG Evaluation | Q001-Q200
Metrics:
  Generation : Token-F1, EM, BLEU-1/4, GLEU, ROUGE-1/2/L/Lsum, METEOR
  Semantic   : SBERT, BERTScore
  Retrieval  : Precision@5, Recall@5, MRR, NDCG@5, HitRate@5, Avg-Rerank-Score
  Faithfulness: LLM hallucination check
  Relevance  : Context Relevance, Answer Relevance (SBERT)
  Agentic    : Avg iterations, Avg confidence
  LLM Rubric : Weighted S.C.O.P.E, LLM-as-a-Judge
"""

import os, re, json, math, sys, io, string, time
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from bert_score import score as bertscore
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score as nltk_meteor
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

from retrieval.retrieve import retrieve
from retrieval.rerank import rerank_with_scores
from generator.generate import generate_answer

load_dotenv()

# ── clients & models ──────────────────────────────────────────────────────
client = OpenAI(
    base_url=os.getenv("LLAMA_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("LLAMA_API_KEY"),
)
judge_model = os.getenv("LLAMA_MODEL_NAME", "meta-llama/llama-3-8b-instruct")

sbert    = SentenceTransformer("all-MiniLM-L6-v2")
rouge    = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
smoother = SmoothingFunction().method1

# SCOPE dimension weights (Correctness heaviest for medical domain)
SCOPE_WEIGHTS = {"S": 0.20, "C": 0.30, "O": 0.15, "P": 0.25, "E": 0.10}

# Relevance threshold: chunk SBERT sim >= this → treat as relevant
RELEVANCE_THRESH = 0.45

# ─────────────────────────────────────────────────────────────────────────────
# Q001 – Q200
# ─────────────────────────────────────────────────────────────────────────────
EVAL_QA = [
  {
    "id": "Q001",
    "q": "What are the three main anatomical divisions of the larynx?",
    "a": "The larynx is anatomically divided into the supraglottic larynx, the glottis, and the subglottis .",
    "category": "diagnosis",
    "difficulty": "simple"
  },
  {
    "id": "Q002",
    "q": "What are the major goals when treating carcinoma of the larynx?",
    "a": "The major goals include maximizing cure, preserving the function of the larynx, preserving voice quality, maintaining good quality of life, and palliating symptoms in incurable disease .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q003",
    "q": "Which threshold of distant recurrence risk is often used to recommend systemic adjuvant chemotherapy in early stage breast cancer?",
    "a": "A risk of distant recurrence of greater than or equal to 10% is often used as the threshold for recommending systemic adjuvant chemotherapy .",
    "category": "treatment",
    "difficulty": "complex"
  },
  {
    "id": "Q004",
    "q": "What do the 'ABCDEs' of early malignant melanoma diagnosis stand for?",
    "a": "A denotes lesion asymmetry, B border irregularity, C color variegation, D diameter greater than 6 mm, and E a lesion that is elevating, evolving or enlarging .",
    "category": "diagnosis",
    "difficulty": "simple"
  },
  {
    "id": "Q005",
    "q": "What is Xeroderma pigmentosum?",
    "a": "It is a rare autosomal recessive disease characterized by photophobia, severe sun sensitivity, defective DNA excision repair, and an incredibly high rate of skin and eye malignancies .",
    "category": "epidemiology",
    "difficulty": "moderate"
  },
  {
    "id": "Q006",
    "q": "Which imaging modalities are used to evaluate the extent of local involvement and regional nodal basins for high-risk squamous cell carcinoma (SCC)?",
    "a": "Magnetic resonance imaging can be used to evaluate local involvement, and an ultrasound can be used for the regional nodal basin .",
    "category": "staging",
    "difficulty": "moderate"
  },
  {
    "id": "Q007",
    "q": "Activating point mutations in which gene are seen in 85% to 95% of patients with gastrointestinal stromal tumors (GISTs)?",
    "a": "Activating point mutations in c-KIT are seen in approximately 85% to 95% of patients with GISTs .",
    "category": "biomarker",
    "difficulty": "complex"
  },
  {
    "id": "Q008",
    "q": "What is the most common presenting symptom for head and neck cancers?",
    "a": "The most common presenting symptom for head and neck cancers is pain .",
    "category": "diagnosis",
    "difficulty": "simple"
  },
  {
    "id": "Q009",
    "q": "What pathological features at the primary tumor site are associated with worse prognosis in head and neck cancers?",
    "a": "Depth of invasion, perineural invasion, perivascular invasion, and lymph node extracapsular spread are associated with worse prognosis .",
    "category": "prognosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q010",
    "q": "What is the mainstay of treatment for early-stage head and neck cancer?",
    "a": "The mainstay for treatment of early-stage head and neck cancer is single modality therapy, either surgery or radiation therapy .",
    "category": "treatment",
    "difficulty": "simple"
  },
  {
    "id": "Q011",
    "q": "Why is it necessary to rule out a parotid malignancy in a patient with sudden facial nerve paralysis?",
    "a": "Because Bell’s palsy is a diagnosis of exclusion, and sudden facial nerve paralysis could indicate a parotid malignancy .",
    "category": "diagnosis",
    "difficulty": "complex"
  },
  {
    "id": "Q012",
    "q": "What neurological symptoms comprise Pancoast syndrome in lung cancer patients?",
    "a": "Pancoast syndrome includes shoulder and arm pain, Horner syndrome, and weakness or atrophy of the hand muscles .",
    "category": "diagnosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q013",
    "q": "How common are paraneoplastic syndromes in lung cancer patients?",
    "a": "Paraneoplastic syndromes are found in 10% of patients with lung cancer, most commonly in those with small cell lung cancer (SCLC) .",
    "category": "epidemiology",
    "difficulty": "simple"
  },
  {
    "id": "Q014",
    "q": "Which specific paraneoplastic symptom is strongly suggestive of a thymoma?",
    "a": "Myasthenia gravis is strongly suggestive of a thymoma .",
    "category": "diagnosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q015",
    "q": "What environmental and dietary risk factors are associated with esophageal squamous cell carcinoma?",
    "a": "Risk factors include diets deficient in vitamins A, C, riboflavin, and protein, excessive nitrates, nitrosamines, and fungal contamination of foodstuffs producing aflatoxin .",
    "category": "epidemiology",
    "difficulty": "complex"
  },
  {
    "id": "Q016",
    "q": "What physical sign usually indicates friability of an esophageal tumor or invasion into major vessels?",
    "a": "Hematemesis and melena usually indicate friability of the tumor or invasion into major vessels .",
    "category": "diagnosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q017",
    "q": "What is the recommended initial evaluation for patients suspected of having gastric adenocarcinoma?",
    "a": "The evaluation includes a complete history, physical examination, complete blood cell count, chemistry/liver tests, and a CT of the chest, abdomen, and pelvis .",
    "category": "staging",
    "difficulty": "moderate"
  },
  {
    "id": "Q018",
    "q": "What defines the threshold for recommending palliative therapy in gastric cancer?",
    "a": "Palliative therapy is considered in patients with systemic disease (AJCC stage IV), depending on their symptoms and functional status .",
    "category": "treatment",
    "difficulty": "simple"
  },
  {
    "id": "Q019",
    "q": "What tests are included in the clinical evaluation of carcinoma of the colon?",
    "a": "Evaluation includes colonoscopy and biopsy, chest radiograph, complete blood cell count, CEA determination, urinalysis, and liver function tests .",
    "category": "diagnosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q020",
    "q": "What is the primary determinant of 5-year survival in colon cancer?",
    "a": "Nodal involvement is the primary determinant of 5-year survival .",
    "category": "prognosis",
    "difficulty": "simple"
  },
  {
    "id": "Q021",
    "q": "Which congenital disease is associated with an increased risk of cholangiocarcinoma?",
    "a": "Caroli disease, a congenital disease characterized by multiple intrahepatic biliary cysts, is associated with an increased risk .",
    "category": "epidemiology",
    "difficulty": "moderate"
  },
  {
    "id": "Q022",
    "q": "Which genetic syndrome is strongly associated with clear cell renal cell carcinomas?",
    "a": "Von Hippel-Lindau (VHL) disease is strongly associated with bilateral clear cell renal cell carcinomas .",
    "category": "epidemiology",
    "difficulty": "moderate"
  },
  {
    "id": "Q023",
    "q": "What is the classic triad of symptoms for renal cell carcinoma, and how often does it occur?",
    "a": "The classic 'too late' triad consists of hematuria, abdominal mass, and flank pain, occurring in approximately 19% of patients .",
    "category": "diagnosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q024",
    "q": "In what percentage of ovarian cancer cases is the CA-125 marker elevated?",
    "a": "CA-125 is elevated in approximately 80% of cases of ovarian cancer .",
    "category": "biomarker",
    "difficulty": "simple"
  },
  {
    "id": "Q025",
    "q": "What is the strongest independent predictor of prolonged survival in ovarian cancer?",
    "a": "The strongest independent predictor of prolonged survival is the absence of residual tumor after surgery .",
    "category": "prognosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q026",
    "q": "When is cervical cancer staging typically performed?",
    "a": "Unlike endometrial and ovarian cancer, cervical cancer staging is performed before treatment planning based on clinical examination, not at the time of diagnosis .",
    "category": "staging",
    "difficulty": "complex"
  },
  {
    "id": "Q027",
    "q": "How quickly are the majority of recurrent cervical cancer cases diagnosed?",
    "a": "More than 50% are diagnosed within 1 year after primary treatment is completed, and 75% within 2 years .",
    "category": "prognosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q028",
    "q": "What is the typical radiographic appearance of a spinal hemangioma?",
    "a": "They have a characteristic honeycomb appearance on plain films due to linear reactive calcification around radiolucent vascular tissue .",
    "category": "diagnosis",
    "difficulty": "simple"
  },
  {
    "id": "Q029",
    "q": "What is the therapeutic goal when evaluating patients with Cancer of Unknown Primary (CUP)?",
    "a": "The goal is to identify those tumor types for which a cure or effective specific therapy is an option .",
    "category": "general",
    "difficulty": "moderate"
  },
  {
    "id": "Q030",
    "q": "What imaging modality is recommended for women with isolated axillary lymph node metastases and suspected occult breast carcinoma?",
    "a": "Magnetic Resonance Imaging (MRI) of the breast is recommended, as it can detect tumors in up to 75% of these patients .",
    "category": "investigation",
    "difficulty": "moderate"
  },
  {
    "id": "Q031",
    "q": "What marker aids in the diagnosis of hepatocellular cancer in patients presenting with liver-only metastatic disease?",
    "a": "Hep-par-1 is a hepatocellular carcinoma marker that aids in its diagnosis .",
    "category": "biomarker",
    "difficulty": "simple"
  },
  {
    "id": "Q032",
    "q": "What does 'neoadjuvant treatment' refer to in oncology?",
    "a": "Neoadjuvant treatment is therapy given in the preoperative or perioperative period, often to improve resectability and organ preservation .",
    "category": "treatment",
    "difficulty": "simple"
  },
  {
    "id": "Q033",
    "q": "What proportion of patients with any stage of cancer experience significant pain?",
    "a": "As many as 60% of patients with any stage of disease experience significant pain .",
    "category": "side_effects",
    "difficulty": "simple"
  },
  {
    "id": "Q034",
    "q": "How does early detection affect observed survival rates via lead-time bias?",
    "a": "Patients appear to live longer from diagnosis because the cancer was detected earlier, rather than because the treatment improved survival .",
    "category": "epidemiology",
    "difficulty": "complex"
  },
  {
    "id": "Q035",
    "q": "What enzyme helps cancer cells maintain immortality by replenishing chromosome ends?",
    "a": "Telomerase replenishes the telomeres of cancer cells, allowing them to remain immortal .",
    "category": "mechanism",
    "difficulty": "moderate"
  },
  {
    "id": "Q036",
    "q": "When is chemotherapy generally postponed regarding patient status?",
    "a": "It is postponed if the patient has an infection, persistent toxicity, or a Karnofsky performance status of less than 50%, unless the tumor is highly aggressive but responsive .",
    "category": "treatment",
    "difficulty": "complex"
  },
  {
    "id": "Q037",
    "q": "What does 'palliation' mean in the context of incurable cancer treatment?",
    "a": "Palliation means improvement of symptoms and function, not necessarily the reduction in size of an asymptomatic lesion .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q038",
    "q": "What percentage of white plaques (leukoplakia) in the head and neck may be cancer in situ?",
    "a": "Five to 10% of white plaques (leukoplakia) may be cancer in situ .",
    "category": "diagnosis",
    "difficulty": "simple"
  },
  {
    "id": "Q039",
    "q": "Which cranial nerve is usually the first to be affected by nasopharyngeal tumors spreading to the cavernous sinus?",
    "a": "Cranial nerve VI is usually the first to be affected, resulting in lateral rectus muscle paresis .",
    "category": "diagnosis",
    "difficulty": "complex"
  },
  {
    "id": "Q040",
    "q": "In lung cancer staging, what is the significance of a mediastinal lymph node measuring 1.5 cm on a CT scan?",
    "a": "Mediastinal lymph nodes are generally considered abnormal when larger than 1.5 cm in diameter .",
    "category": "staging",
    "difficulty": "moderate"
  },
  {
    "id": "Q041",
    "q": "What percentage of patients with Small Cell Lung Cancer (SCLC) have neurologically asymptomatic brain metastases?",
    "a": "SCLC is associated with a 10% incidence of neurologically asymptomatic brain metastases .",
    "category": "epidemiology",
    "difficulty": "moderate"
  },
  {
    "id": "Q042",
    "q": "What role does involuntary weight loss play as a prognostic factor in cancer?",
    "a": "Involuntary weight loss of 5% or more is an independent and negative prognostic factor .",
    "category": "prognosis",
    "difficulty": "simple"
  },
  {
    "id": "Q043",
    "q": "What are the three grave prognostic signs for gastric cancer post-surgery?",
    "a": "The three grave signs are serosal involvement, nodal involvement, and tumor at the line of resection .",
    "category": "prognosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q044",
    "q": "What allelic loss in colorectal cancer indicates a significantly worse prognosis?",
    "a": "An allele loss of chromosome 18q is significantly associated with a worse prognosis .",
    "category": "biomarker",
    "difficulty": "complex"
  },
  {
    "id": "Q045",
    "q": "What percentage of patients treated with mastectomy or RT for breast cancer have evidence of tumor at autopsy regardless of cause of death?",
    "a": "75% to 85% of patients with a history of breast cancer have evidence of the tumor at autopsy .",
    "category": "prognosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q046",
    "q": "What tumor markers are useful for monitoring the response to therapy in advanced breast cancer?",
    "a": "Blood CEA and CA 27.29 (CA 15-3) levels may be useful to follow response to treatment .",
    "category": "biomarker",
    "difficulty": "simple"
  },
  {
    "id": "Q047",
    "q": "What is the typical presentation of testicular cancer?",
    "a": "The most common symptom is a painless enlargement of the testis, usually noticed during bathing or after minor trauma .",
    "category": "diagnosis",
    "difficulty": "simple"
  },
  {
    "id": "Q048",
    "q": "In renal cell carcinoma, which histologic pattern is associated with a poor prognosis?",
    "a": "Sarcomatous patterns of RCC have a poor prognosis .",
    "category": "pathology",
    "difficulty": "moderate"
  },
  {
    "id": "Q049",
    "q": "What is the risk of progression for Carcinoma In Situ (CIS) of the bladder if left untreated?",
    "a": "CIS progresses to invasive carcinoma in 80% of patients within 10 years of diagnosis .",
    "category": "prognosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q050",
    "q": "What does a high nuclear grade (Fuhrman's system) indicate in renal cell carcinoma?",
    "a": "Nuclear grade correlates with survival across all tumor stages, with higher grades generally indicating worse prognosis .",
    "category": "pathology",
    "difficulty": "complex"
  },
  {
    "id": "Q051",
    "q": "How does prostate specific antigen (PSA) levels help in diagnosing prostate cancer?",
    "a": "PSA acts as a marker unique to the prostate, significantly augmenting the yield of Digital Rectal Examination (DRE) in diagnosing prostate cancer .",
    "category": "biomarker",
    "difficulty": "moderate"
  },
  {
    "id": "Q052",
    "q": "What type of headaches are characteristically associated with fast-growing CNS tumors?",
    "a": "Headaches that are deep, dull, worse on arising in the morning, and exacerbated by straining or lifting .",
    "category": "diagnosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q053",
    "q": "What symptom is typically associated with a tumor in the parietal lobe?",
    "a": "A supratentorial lobar tumor in the parietal lobe typically presents with hemineglect .",
    "category": "diagnosis",
    "difficulty": "complex"
  },
  {
    "id": "Q054",
    "q": "What proportion of palpable 'cold' thyroid nodules prove to be cancer?",
    "a": "Only about 10% of cold thyroid nodules prove to be cancer .",
    "category": "diagnosis",
    "difficulty": "simple"
  },
  {
    "id": "Q055",
    "q": "What triad of symptoms is highly suggestive of sellar (pituitary) metastases?",
    "a": "The triad of headache, extraocular nerve palsy, and diabetes insipidus .",
    "category": "diagnosis",
    "difficulty": "complex"
  },
  {
    "id": "Q056",
    "q": "Which immunosuppressive drug is associated with cutaneous SCC and lymphoma?",
    "a": "Cyclosporine is associated with an increased risk of cutaneous SCC and lymphoma .",
    "category": "epidemiology",
    "difficulty": "moderate"
  },
  {
    "id": "Q057",
    "q": "What defines the Stewart-Treves syndrome?",
    "a": "It is the development of lymphangiosarcoma in patients with prolonged postmastectomy arm edema .",
    "category": "epidemiology",
    "difficulty": "moderate"
  },
  {
    "id": "Q058",
    "q": "What is the most important factor in predicting the behavior of a soft tissue sarcoma?",
    "a": "The degree of cellular differentiation (grade) and the amount of necrosis within the tumor are the most important factors .",
    "category": "prognosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q059",
    "q": "What indicates poor survival in neuroblastoma stages III and IV?",
    "a": "Patients with stage III and IV disease who have amplification of the N-myc gene do worse .",
    "category": "biomarker",
    "difficulty": "complex"
  },
  {
    "id": "Q060",
    "q": "What is the typical presentation of a Wilms tumor?",
    "a": "The most common finding is a palpable abdominal mass, alongside an enlarged abdomen and painless hematuria .",
    "category": "diagnosis",
    "difficulty": "simple"
  },
  {
    "id": "Q061",
    "q": "Which chemotherapy regimen is most commonly given for rhabdomyosarcoma?",
    "a": "The VAC (vincristine, actinomycin D, cyclophosphamide) regimen is most commonly given .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q062",
    "q": "What does a sudden appearance of multiple seborrheic keratoses (Leser-Trélat sign) typically indicate?",
    "a": "It is usually a rare paraneoplastic manifestation of gastrointestinal cancer .",
    "category": "diagnosis",
    "difficulty": "complex"
  },
  {
    "id": "Q063",
    "q": "What is the main objective of primary prevention in oncology?",
    "a": "The objectives include reduction of cancer incidence, reduction of adverse effects of treatment, and reduction of mortality .",
    "category": "general",
    "difficulty": "simple"
  },
  {
    "id": "Q064",
    "q": "How is Disease Free Survival (DFS) defined?",
    "a": "DFS is the time period during which the patient is alive and healthy and has no signs of disease after previous successful treatment of the primary tumor .",
    "category": "prognosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q065",
    "q": "When is curative chemotherapy used?",
    "a": "It is used in highly chemosensitive diseases like childhood tumors, hematological malignancies, and solid tumors like testicular tumors and choriocarcinomas .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q066",
    "q": "What is the goal of symptomatic treatment?",
    "a": "The goal is the liquidation and reduction of symptoms produced by the tumor itself, without impairing the quality of life .",
    "category": "treatment",
    "difficulty": "simple"
  },
  {
    "id": "Q067",
    "q": "Why is 'tailoring' or selective targeting used in modern chemotherapy?",
    "a": "To identify specific groups of patients who will most profit from chosen therapy based on prognostic and predictive factors, minimizing non-selective toxicity .",
    "category": "treatment",
    "difficulty": "complex"
  },
  {
    "id": "Q068",
    "q": "What are the common visual symptoms when a nasopharyngeal carcinoma ingrows into the orbit?",
    "a": "It causes protrusion of the eyeball and double vision .",
    "category": "diagnosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q069",
    "q": "Why is cetuximab used in head and neck cancer treatment?",
    "a": "Cetuximab is an epidermal growth factor receptor (EGFR) inhibitor used in combination with radiotherapy for locally advanced disease .",
    "category": "treatment",
    "difficulty": "complex"
  },
  {
    "id": "Q070",
    "q": "What determines the indication of adjuvant treatment after surgery in head and neck cancer?",
    "a": "It depends on the extent and location of the tumor, lymph node involvement, and other risk factors .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q071",
    "q": "Which type of thyroid cancer requires adjuvant radiotherapy?",
    "a": "Adjuvant radiotherapy is indicated in well-differentiated tumors with insufficient/no accumulation of radioactive iodine, or when the tumor has infiltrated surrounding connective tissue .",
    "category": "treatment",
    "difficulty": "complex"
  },
  {
    "id": "Q072",
    "q": "What is the significance of the BRCA1 and BRCA2 genes in breast cancer?",
    "a": "Inherited mutations in these genes significantly increase the risk of developing malignant tumors in the breast and ovaries .",
    "category": "epidemiology",
    "difficulty": "simple"
  },
  {
    "id": "Q073",
    "q": "What paraneoplastic neurological symptoms can manifest in small cell lung cancer?",
    "a": "Central symptoms like brain atrophy and dementia, and peripheral manifestations like neuropathy and myasthenia .",
    "category": "diagnosis",
    "difficulty": "complex"
  },
  {
    "id": "Q074",
    "q": "What is the main treatment method for locally advanced non-small cell lung cancer (NSCLC)?",
    "a": "Radiotherapy, normally performed via external radiation, combined with concomitant chemotherapy is the main radical approach .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q075",
    "q": "What are the targeted therapy options for EGFR mutated NSCLC?",
    "a": "Gefitinib, erlotinib, afatinib, and osimertinib are used as targeted treatments .",
    "category": "treatment",
    "difficulty": "complex"
  },
  {
    "id": "Q076",
    "q": "What defines a Krukenberg tumor?",
    "a": "It is a metastatic spread of cancer, often from the stomach, to the ovary .",
    "category": "pathology",
    "difficulty": "moderate"
  },
  {
    "id": "Q077",
    "q": "What is the FLOT regimen used for?",
    "a": "It is a perioperative chemotherapy regimen (4 cycles before and 4 cycles after surgery) used for localized gastric cancer .",
    "category": "treatment",
    "difficulty": "complex"
  },
  {
    "id": "Q078",
    "q": "What are common symptoms of right-sided colon tumors?",
    "a": "Indeterminate pain, abdominal pressure, fatigue, and weakness due to manifestations of anemia from chronic, occult blood loss .",
    "category": "diagnosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q079",
    "q": "What does a 5-year survival rate of ~40% indicate in colorectal cancer?",
    "a": "It generally indicates diseases with nodal positivity (stage III) .",
    "category": "prognosis",
    "difficulty": "simple"
  },
  {
    "id": "Q080",
    "q": "What is the basic surgical treatment for testicular cancer?",
    "a": "The basic treatment is always radical inguinal orchiectomy, performed within 24-48 hours after diagnosis .",
    "category": "treatment",
    "difficulty": "simple"
  },
  {
    "id": "Q081",
    "q": "What factors determine the prognosis of prostate cancer?",
    "a": "Prognosis depends on the extent of the disease (TNM), Gleason score (GS), pretreatment PSA levels, patient age, and general condition .",
    "category": "prognosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q082",
    "q": "Why is TNM classification not applicable to brain tumors?",
    "a": "Because brain tumors typically do not spread to regional lymph nodes or metastasize distantly in the traditional sense, making TNM irrelevant .",
    "category": "staging",
    "difficulty": "moderate"
  },
  {
    "id": "Q083",
    "q": "What is the treatment strategy for generalized malignant melanoma with a BRAF V600 mutation?",
    "a": "They can be treated with BRAF inhibitors (vemurafenib, dabrafenib) or a combination of BRAF and MEK inhibitors .",
    "category": "treatment",
    "difficulty": "complex"
  },
  {
    "id": "Q084",
    "q": "What is the most frequently used tumor marker for epithelial ovarian tumors?",
    "a": "Ca 125 is elevated in 95% of malignant ovarian epithelial tumors .",
    "category": "biomarker",
    "difficulty": "simple"
  },
  {
    "id": "Q085",
    "q": "What characterizes the 'watch and wait' strategy in advanced follicular lymphoma?",
    "a": "Avoiding chemotherapy-based treatment until the patient has significant symptoms .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q086",
    "q": "What cytogenetic translocation is characteristic of Synovial Sarcoma?",
    "a": "The translocation t(x;18) (p11;q11) is characteristic of Synovial Sarcoma .",
    "category": "biomarker",
    "difficulty": "complex"
  },
  {
    "id": "Q087",
    "q": "How does early detection of cancer influence the concept of lead-time bias?",
    "a": "Patients appear to survive longer from diagnosis simply because the cancer was found earlier, not because the treatment extended their ultimate lifespan .",
    "category": "epidemiology",
    "difficulty": "complex"
  },
  {
    "id": "Q088",
    "q": "What tumor marker is useful in the diagnosis and monitoring of pancreatic cancer?",
    "a": "CA 19-9 is useful for pancreatic cancer, with a 70% specificity and 90% sensitivity .",
    "category": "biomarker",
    "difficulty": "simple"
  },
  {
    "id": "Q089",
    "q": "Which environmental carcinogen exposure is linked to the development of angiosarcoma of the liver?",
    "a": "Exposure to Thorotrast or polyvinyl chloride is linked to angiosarcoma .",
    "category": "epidemiology",
    "difficulty": "moderate"
  },
  {
    "id": "Q090",
    "q": "What are the common symptoms of spinal cord tumors?",
    "a": "Symptoms include compression signs like back pain, spastic paraparesis, sensory loss below the tumor, and bowel/bladder dysfunction .",
    "category": "diagnosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q091",
    "q": "What is the role of the p53 gene in cancer development?",
    "a": "p53 acts as a tumor suppressor gene; its mutation or loss allows for uncontrolled cell proliferation and avoidance of apoptosis .",
    "category": "mechanism",
    "difficulty": "moderate"
  },
  {
    "id": "Q092",
    "q": "How does alcohol act synergistically with tobacco in head and neck cancer?",
    "a": "Heavy smoking combined with excess alcohol consumption results in over 35 times the risk of oral cancer compared to a person who does neither .",
    "category": "epidemiology",
    "difficulty": "moderate"
  },
  {
    "id": "Q093",
    "q": "What are the main roles of surgery in the management of cancer?",
    "a": "Diagnosis and staging, curative surgery, palliative surgery, surgery for metastatic disease, and prophylactic surgery .",
    "category": "treatment",
    "difficulty": "simple"
  },
  {
    "id": "Q094",
    "q": "What characterizes the presentation of carcinoid tumors of the appendix?",
    "a": "They are typically diagnosed incidentally after an appendectomy .",
    "category": "diagnosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q095",
    "q": "What is the main driver of global cervical cancer disparities in mortality?",
    "a": "Lack of access to screening and timely treatment in low- and middle-income countries drives vast mortality disparities .",
    "category": "epidemiology",
    "difficulty": "complex"
  },
  {
    "id": "Q096",
    "q": "What is 'liquid biopsy' in oncology?",
    "a": "It is the analysis of tumor-derived products, like circulating cell-free tumor DNA (ctDNA) or circulating tumor cells, detectable in blood or other body fluids .",
    "category": "investigation",
    "difficulty": "simple"
  },
  {
    "id": "Q097",
    "q": "Why is the use of systemic therapy often deferred in indolent asymptomatic cancers?",
    "a": "To delay exposing patients to the potential side effects of chemotherapy until the disease progresses and becomes symptomatic .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q098",
    "q": "Which genetic syndrome predisposes individuals to medullary thyroid cancer?",
    "a": "Germline mutations in the RET proto-oncogene, often seen in Multiple Endocrine Neoplasia (MEN) syndromes, predispose to medullary carcinoma .",
    "category": "biomarker",
    "difficulty": "complex"
  },
  {
    "id": "Q099",
    "q": "What is the typical clinical presentation of a rhabdomyosarcoma in children?",
    "a": "It often presents as a painless, enlarging mass depending on the location, such as the orbit, head and neck, or genitourinary tract .",
    "category": "diagnosis",
    "difficulty": "simple"
  },
  {
    "id": "Q100",
    "q": "What is the significance of the Philadelphia chromosome?",
    "a": "It is a genetic marker (chromosome abnormality) primarily useful for the diagnosis and targeted treatment of chronic myeloid leukemia .",
    "category": "biomarker",
    "difficulty": "moderate"
  },
  {
    "id": "Q101",
    "q": "What is the purpose of metronomic chemotherapy?",
    "a": "It involves the continuous or frequent administration of low doses of cytotoxic drugs to achieve prolonged disease control with minimal side effects .",
    "category": "treatment",
    "difficulty": "complex"
  },
  {
    "id": "Q102",
    "q": "What role does the Epstein-Barr virus (EBV) play in oncology?",
    "a": "EBV infection is strongly linked to the development of endemic nasopharyngeal carcinoma and Burkitt's lymphoma .",
    "category": "epidemiology",
    "difficulty": "simple"
  },
  {
    "id": "Q103",
    "q": "What is length bias in cancer screening?",
    "a": "Length bias occurs because slow-growing, less aggressive tumors are more likely to be detected by screening tests compared to fast-growing, aggressive tumors .",
    "category": "epidemiology",
    "difficulty": "complex"
  },
  {
    "id": "Q104",
    "q": "Which biological marker is essential to evaluate in breast cancer for predicting response to targeted therapies like Trastuzumab?",
    "a": "The HER2/neu (c-erbB-2) receptor expression must be evaluated .",
    "category": "biomarker",
    "difficulty": "simple"
  },
  {
    "id": "Q105",
    "q": "How does systemic inflammatory response, measured by CRP and albumin (Glasgow Prognostic Score), impact cancer prognosis?",
    "a": "A high Glasgow Prognostic Score (elevated systemic inflammation) generally correlates with poorer overall patient survival .",
    "category": "prognosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q106",
    "q": "What defines 'disease-free interval' (DFI) in cancer prognosis?",
    "a": "DFI is the time elapsed from the resection of the primary tumor to the detection of metastases; a longer DFI usually indicates a better prognosis .",
    "category": "prognosis",
    "difficulty": "simple"
  },
  {
    "id": "Q107",
    "q": "What is the typical initial imaging method for a suspected pediatric neck mass?",
    "a": "Detailed ultrasonography of the entire neck region is the preferred initial test because it is painless and does not require anesthesia .",
    "category": "investigation",
    "difficulty": "simple"
  },
  {
    "id": "Q108",
    "q": "What causes the 'watery diarrhea-hypokalemia-achlorhydria' (WDHA) syndrome?",
    "a": "It is a paraneoplastic syndrome caused by the production of vasoactive intestinal peptide (VIP), often seen in neuroendocrine tumors like Verner-Morrison syndrome .",
    "category": "diagnosis",
    "difficulty": "complex"
  },
  {
    "id": "Q109",
    "q": "What is the primary risk factor for developing mesothelioma?",
    "a": "Exposure to asbestos is the primary recognized risk factor for developing mesothelioma .",
    "category": "epidemiology",
    "difficulty": "simple"
  },
  {
    "id": "Q110",
    "q": "What is an oncocytoma of the salivary gland?",
    "a": "It is a benign, circumscribed mass composed of large oncocytic epithelial cells with abundant eosinophilic granular cytoplasm .",
    "category": "pathology",
    "difficulty": "moderate"
  },
  {
    "id": "Q111",
    "q": "What is 'tumor budding' and its significance in squamous cell carcinoma?",
    "a": "Tumor budding refers to single cells or small clusters of cells at the advancing invasive edge, and it predicts distant metastasis and poorer survival .",
    "category": "prognosis",
    "difficulty": "complex"
  },
  {
    "id": "Q112",
    "q": "How does human papillomavirus (HPV) status affect oropharyngeal cancer prognosis?",
    "a": "Patients with HPV-positive oropharyngeal cancer generally have a better prognosis and higher survival rates than those with HPV-negative cancers .",
    "category": "prognosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q113",
    "q": "What is the clinical presentation of a solitary fibrous tumor (SFT) in the head and neck?",
    "a": "Patients typically present with a painless, slow-growing mass that strongly enhances with contrast on CT imaging .",
    "category": "diagnosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q114",
    "q": "What kind of diet is associated with an increased risk of colorectal cancer?",
    "a": "A diet high in fat, red meats, sausages, and low in fiber is associated with an increased risk .",
    "category": "epidemiology",
    "difficulty": "simple"
  },
  {
    "id": "Q115",
    "q": "In pediatric germ-cell tumors, why are ovarian biopsies generally contraindicated?",
    "a": "Biopsies are avoided due to the high oncological risk of causing peritoneal spread of malignant cells .",
    "category": "treatment",
    "difficulty": "complex"
  },
  {
    "id": "Q116",
    "q": "Which cancer types most commonly metastasize to the brain?",
    "a": "Primary lung cancer, breast cancer, occult renal cell cancer, and melanoma are the most frequent origins for brain metastases .",
    "category": "general",
    "difficulty": "moderate"
  },
  {
    "id": "Q117",
    "q": "What defines a 'sporadic' cancer?",
    "a": "A sporadic cancer develops due to spontaneous, random somatic mutations or replication errors rather than from an inherited germline mutation .",
    "category": "mechanism",
    "difficulty": "simple"
  },
  {
    "id": "Q118",
    "q": "What imaging study is most appropriate for assessing local tumor extent and depth of invasion in rectal cancer?",
    "a": "MRI of the pelvis or endorectal ultrasound is used to accurately assess local tumor extent and depth of invasion .",
    "category": "investigation",
    "difficulty": "moderate"
  },
  {
    "id": "Q119",
    "q": "How do alkylating agents contribute to carcinogenesis?",
    "a": "Alkylating agents are genotoxic carcinogens that transfer alkyl groups to specific sites on DNA bases, causing somatic mutations relevant to cancer pathogenesis .",
    "category": "mechanism",
    "difficulty": "complex"
  },
  {
    "id": "Q120",
    "q": "What is the role of sentinel lymph node biopsy in breast cancer management?",
    "a": "A negative axillary sentinel lymph node biopsy accurately estimates prognosis and allows patients to avoid the morbidity of a full axillary dissection .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q121",
    "q": "What is the most common presenting sign of newly diagnosed breast cancer?",
    "a": "More than 85% of newly diagnosed breast cancers are detected as a lump in the breast, often accompanied by a thickening felt by the patient .",
    "category": "diagnosis",
    "difficulty": "simple"
  },
  {
    "id": "Q122",
    "q": "How is Lobular Carcinoma In Situ (LCIS) defined pathologically?",
    "a": "LCIS is a proliferative lesion defined by the colonization of the terminal duct lobular unit by small discohesive cells, expanding the structures and obliterating the lumina, without invading the basement membrane .",
    "category": "pathology",
    "difficulty": "moderate"
  },
  {
    "id": "Q123",
    "q": "What are the three categories of endocrine responsiveness in early breast cancer?",
    "a": "The categories are highly endocrine responsive (high expression of both ER and PR), incompletely endocrine responsive (lower expression), and endocrine nonresponsive (complete absence of both) .",
    "category": "biomarker",
    "difficulty": "moderate"
  },
  {
    "id": "Q124",
    "q": "For recurrent or metastatic breast cancer patients who have failed anthracycline-based therapies, which drug class is typically preferred?",
    "a": "Taxane-based regimens are typically preferred, and both single-agent and combination regimens can be selected for the first-line treatment in these patients .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q125",
    "q": "What is the approximate male-to-female incidence ratio for head and neck cancers?",
    "a": "The incidence of squamous cell carcinoma in the head and neck is significantly higher in male patients, with a male-to-female ratio of approximately 3:1 to 4:1 .",
    "category": "epidemiology",
    "difficulty": "simple"
  },
  {
    "id": "Q126",
    "q": "In countries like India, what is the primary risk factor for developing oral cavity cancer?",
    "a": "In India, oral cavity cancer is mainly caused by betel quid chewing .",
    "category": "epidemiology",
    "difficulty": "simple"
  },
  {
    "id": "Q127",
    "q": "How do immune checkpoint molecules influence the tumor microenvironment?",
    "a": "Immune checkpoint molecules are key modulators that activate inhibitory signaling pathways to dampen interactions like CTLA-4/B7 and PD-1/PD-L1, thus blocking anti-tumor T cell immune responses .",
    "category": "mechanism",
    "difficulty": "complex"
  },
  {
    "id": "Q128",
    "q": "What are some significant adverse events associated with non-specific immunostimulation therapies?",
    "a": "Systemic and bystander immune-related adverse events (irAEs) such as autoimmune and inflammatory toxicities, like colitis, are reported, though most are reversible if treated promptly .",
    "category": "side_effects",
    "difficulty": "moderate"
  },
  {
    "id": "Q129",
    "q": "What is the lifetime risk of developing ovarian cancer for carriers of BRCA mutations?",
    "a": "Carriers of BRCA mutations have an estimated 10% to 40% lifetime risk of developing ovarian cancer .",
    "category": "epidemiology",
    "difficulty": "simple"
  },
  {
    "id": "Q130",
    "q": "What is the mechanism of action of PARP inhibitors in treating BRCA-mutated ovarian cancer?",
    "a": "PARP inhibitors generate double-strand DNA breaks that require functional BRCA1 and BRCA2 proteins for repair, effectively targeting and destroying cancer cells with dysfunctional BRCA proteins .",
    "category": "mechanism",
    "difficulty": "complex"
  },
  {
    "id": "Q131",
    "q": "Why is upfront surgical resection recommended for Wilms tumor in pediatric patients less than 6 months of age?",
    "a": "Upfront resection is recommended due to the relatively higher incidence of congenital mesoblastic nephroma and rhabdoid tumors in this specific age group .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q132",
    "q": "Which stages and genetic markers define high-risk neuroblastoma in patients over 18 months (547 days) of age?",
    "a": "Patients older than 547 days with Stages 1, 2, or 3 and MYCN amplification, or Stage 4 disease regardless of MYCN status, are categorized as high-risk .",
    "category": "staging",
    "difficulty": "complex"
  },
  {
    "id": "Q133",
    "q": "Why do patients with gastric adenocarcinoma often present with a poor prognosis?",
    "a": "Gastric adenocarcinoma has a propensity for early dissemination, and the majority of patients present with advanced disease because early symptoms are often inconspicuous .",
    "category": "prognosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q134",
    "q": "What is the purpose of consolidating therapy with radiation in oncology?",
    "a": "Consolidation with radiation aims to reduce local recurrences as part of the initial treatment plan, although it may not affect overall survival if distant metastases are present .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q135",
    "q": "Which cranial nerve is typically the first to be affected by nasopharyngeal carcinoma spreading to the cavernous sinus?",
    "a": "Cranial nerve VI is usually the first affected, resulting in lateral rectus muscle paresis .",
    "category": "diagnosis",
    "difficulty": "complex"
  },
  {
    "id": "Q136",
    "q": "What is the World Health Organization's specific target incidence rate for eliminating cervical cancer?",
    "a": "The WHO goal is to eliminate cervical cancer by reducing the incidence rate to less than 4 per 100,000 .",
    "category": "epidemiology",
    "difficulty": "simple"
  },
  {
    "id": "Q137",
    "q": "What is the projected global increase in patients requiring first-course chemotherapy between 2018 and 2040?",
    "a": "The annual number of patients requiring first-course chemotherapy is projected to increase from 9.8 million to 15.0 million .",
    "category": "epidemiology",
    "difficulty": "moderate"
  },
  {
    "id": "Q138",
    "q": "What proportion of the population in lower- and middle-income countries (LMICs) lacks access to safe and affordable cancer surgery?",
    "a": "Over 90% of the population residing in LMICs lacks access to safe, affordable, and timely surgical care .",
    "category": "epidemiology",
    "difficulty": "simple"
  },
  {
    "id": "Q139",
    "q": "What second-line targeted therapies may be considered for NSCLC patients who progress after first-line treatment?",
    "a": "In patients who have progressed after first-line therapy, erlotinib or gefitinib (tyrosine kinase inhibitors) may be considered if they maintain adequate performance status .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q140",
    "q": "What viral infection is strongly associated with the development of nasopharyngeal carcinoma?",
    "a": "There is a near consistent association between Epstein-Barr virus (EBV) and the development of nasopharyngeal carcinoma .",
    "category": "epidemiology",
    "difficulty": "simple"
  },
  {
    "id": "Q141",
    "q": "In cases of Metastasis of Unknown Origin (MUO), how often does a postmortem examination fail to identify the primary tumor?",
    "a": "Postmortem examinations fail to detect the primary tumor site in up to 25% of MUO cases .",
    "category": "investigation",
    "difficulty": "moderate"
  },
  {
    "id": "Q142",
    "q": "What is the standard preferred therapy for most women with operable breast cancer to preserve the breast?",
    "a": "Breast conservation, defined as complete tumor excision (lumpectomy or wide local excision) followed by whole breast irradiation, should be offered as the preferred therapy .",
    "category": "treatment",
    "difficulty": "simple"
  },
  {
    "id": "Q143",
    "q": "What is the recommended treatment for elderly patients with advanced diffuse large B-cell lymphoma (DLBCL)?",
    "a": "Elderly patients with advanced disease should be treated with the R-CHOP regimen, or if there is a contraindication to anthracyclines, with R-FC or rituximab/ifosfamide/etoposide .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q144",
    "q": "What characteristic genetic fusion is present in Chronic Myelogenous Leukemia (CML)?",
    "a": "CML is defined by the presence of the BCR/ABL chimeric gene, which results from a reciprocal t(9;22) translocation occurring in a hematopoietic stem cell .",
    "category": "biomarker",
    "difficulty": "complex"
  },
  {
    "id": "Q145",
    "q": "According to the CSCO guidelines, which breast cancer patients should be considered for preoperative neoadjuvant treatment?",
    "a": "Candidates include those with large tumor size, positive axillary nodes, HER2-positive, triple-negative tumors, or a desire for breast-conserving surgery blocked by an initially large tumor proportion .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q146",
    "q": "Why do tumors with high frequencies of somatic mutations respond better to immune checkpoint inhibitors?",
    "a": "High mutational burden tumors are enriched in non-synonymous neoantigens, making them potentially more immunogenic and visible to CD8+ T cells .",
    "category": "mechanism",
    "difficulty": "complex"
  },
  {
    "id": "Q147",
    "q": "What long-term complication affects approximately half of pediatric patients treated for salivary gland tumors?",
    "a": "Approximately one half of these patients will present with Frey syndrome in the long term .",
    "category": "side_effects",
    "difficulty": "moderate"
  },
  {
    "id": "Q148",
    "q": "What objective response rate was achieved with the combination of nivolumab and ipilimumab in the phase III CheckMate 067 trial for metastatic melanoma?",
    "a": "The objective response rate was 72.1% with the combination of nivolumab and ipilimumab, compared to 21.3% with ipilimumab alone .",
    "category": "prognosis",
    "difficulty": "complex"
  },
  {
    "id": "Q149",
    "q": "In front-line therapy for NSCLC, what does high PD-L1 expression (>=50%) on tumor cells predict?",
    "a": "It predicts an overall survival benefit for patients treated with pembrolizumab compared to standard cisplatin-based doublet chemotherapy .",
    "category": "biomarker",
    "difficulty": "moderate"
  },
  {
    "id": "Q150",
    "q": "What is a Level I radiotherapy recommendation for a breast cancer patient presenting with diffuse brain metastases?",
    "a": "Whole-brain radiotherapy (WBRT), specifically with hippocampal avoidance, is a Level I recommendation .",
    "category": "treatment",
    "difficulty": "simple"
  },
  {
    "id": "Q151",
    "q": "Which FDA-approved immune checkpoint inhibitor targets mismatch repair-deficient (dMMR) or MSI-H refractory solid tumors agnostically?",
    "a": "Pembrolizumab (an anti-PD-1 antibody) is approved for unresectable or metastatic MSI-H or dMMR solid tumors .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q152",
    "q": "How does the NCCN define an Adolescent and Young Adult (AYA) cancer patient?",
    "a": "An AYA cancer patient is defined as a person 15 to 39 years of age at the time of initial cancer diagnosis .",
    "category": "general",
    "difficulty": "simple"
  },
  {
    "id": "Q153",
    "q": "What are the common late effects of radiation therapy applied to the thyroid gland?",
    "a": "Prior radiation to the thyroid can increase the risk for thyroid disorders such as hypothyroidism, hyperthyroidism, and thyroid cancer .",
    "category": "side_effects",
    "difficulty": "simple"
  },
  {
    "id": "Q154",
    "q": "Adolescent and Young Adult (AYA) patients with germline TP53 mutations (Li-Fraumeni syndrome) are at high risk of developing which bone and soft tissue malignancies?",
    "a": "They are at a higher risk of developing osteosarcoma and rhabdomyosarcoma .",
    "category": "epidemiology",
    "difficulty": "moderate"
  },
  {
    "id": "Q155",
    "q": "What tumor markers should be checked quarterly during the first year of follow-up after radical therapy for colorectal cancer?",
    "a": "Cancer markers CEA, Ca19-9, or Ca72-4 should be monitored .",
    "category": "investigation",
    "difficulty": "moderate"
  },
  {
    "id": "Q156",
    "q": "In the management of prostate cancer, when is the 'watchful waiting' approach utilized?",
    "a": "Watchful waiting aims for palliative treatment and is typically chosen when local treatment is not considered suitable due to the patient's general condition or prognosis .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q157",
    "q": "Why do adolescents and young adults with Acute Lymphoblastic Leukemia (ALL) treated on pediatric protocols often show better survival than those on adult protocols?",
    "a": "The improved outcomes are attributed to the intensive use of nonmyelosuppressive agents (like L-asparaginase), earlier CNS prophylaxis, longer maintenance, and better protocol adherence .",
    "category": "prognosis",
    "difficulty": "complex"
  },
  {
    "id": "Q158",
    "q": "According to the Reese-Ellsworth staging system for retinoblastoma, what features define Group I?",
    "a": "Group I consists of solitary or multiple tumors less than 4 disc diameters in size located at or behind the midplane of the eye .",
    "category": "staging",
    "difficulty": "complex"
  },
  {
    "id": "Q159",
    "q": "Why is a biopsy generally avoided when evaluating a patient for suspected retinoblastoma?",
    "a": "Biopsies are contraindicated and not performed because of the severe risk of seeding tumor cells or causing tumor spread; diagnosis relies on clinical and imaging modalities .",
    "category": "investigation",
    "difficulty": "moderate"
  },
  {
    "id": "Q160",
    "q": "What radiotherapy techniques are preferred for delivering Accelerated Partial Breast Irradiation (APBI)?",
    "a": "Intensity-Modulated Radiation Therapy (IMRT) and interstitial brachytherapy are applied, with IMRT being the preferred technique .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q161",
    "q": "What are the characteristics of an ideal maintenance chemotherapy protocol for advanced breast cancer?",
    "a": "An ideal maintenance protocol uses an effective single-agent therapy (e.g., oral capecitabine) that is relatively low-toxic, convenient for long-term use, and achieves maximal disease control .",
    "category": "treatment",
    "difficulty": "complex"
  },
  {
    "id": "Q162",
    "q": "When is it safe to administer adjuvant chemotherapy to a pregnant patient diagnosed with breast cancer?",
    "a": "Adjuvant chemotherapy should be delayed until at least the second or third trimester, or, if possible, until after the patient's delivery .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q163",
    "q": "What is the purpose of ASCO's Global Guidelines (formerly Resource-Stratified Guidelines)?",
    "a": "They provide evidence-based, economically feasible, and culturally appropriate clinical practice options customized for low- and middle-income nations based on their available health care resources .",
    "category": "general",
    "difficulty": "moderate"
  },
  {
    "id": "Q164",
    "q": "In ER-positive, HER2-negative breast cancer, what does a 21-gene recurrence score (RS) of 31 or greater indicate?",
    "a": "An RS >= 31 indicates high risk of distant local recurrence, and these patients have the largest benefit from the addition of chemotherapy to hormonal therapy .",
    "category": "prognosis",
    "difficulty": "complex"
  },
  {
    "id": "Q165",
    "q": "What does Adoptive T Cell Therapy (ACT) using Tumor Infiltrating Lymphocytes (TILs) entail?",
    "a": "It involves isolating naturally occurring tumor-specific T cells from a patient's tumor, massively expanding them ex vivo, and infusing them back into the lymphodepleted patient .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q166",
    "q": "How do active cancer vaccines aim to alter the tumor microenvironment?",
    "a": "Cancer vaccines aim to stimulate tumor-specific T cells from the naive repertoire and boost existing responses, turning \"cold\" tumors (few TILs) into \"hot\" tumors (many TILs) .",
    "category": "mechanism",
    "difficulty": "complex"
  },
  {
    "id": "Q167",
    "q": "What is the mechanism of tremelimumab in the treatment of hepatocellular carcinoma?",
    "a": "Tremelimumab is a fully human IgG2 monoclonal antibody that acts as an antagonist of CTLA-4 on activated T cells, stimulating T cell activation to enhance tumor eradication .",
    "category": "mechanism",
    "difficulty": "moderate"
  },
  {
    "id": "Q168",
    "q": "Why is the use of PD-1/PD-L1 inhibitors rationalized in the treatment of Merkel cell carcinoma (MCC)?",
    "a": "PD-L1 is expressed in the MCC tumor microenvironment due to chronic antigen presentation from processed viral proteins and UV-induced neoantigens, rendering the tumor sensitive to checkpoint blockade .",
    "category": "treatment",
    "difficulty": "complex"
  },
  {
    "id": "Q169",
    "q": "What are the common clinical presentations of a patient with a paranasal sinus tumor?",
    "a": "Patients often present with epistaxis, unilateral nasal obstruction with discharge, pain and paraesthesia of the cheek, and potentially proptosis or diplopia if the orbit is involved .",
    "category": "clinical_features",
    "difficulty": "moderate"
  },
  {
    "id": "Q170",
    "q": "Which specific virus is fundamentally responsible for driving a high fraction of oropharyngeal squamous cell carcinomas?",
    "a": "Transcriptionally-active high-risk human papillomavirus (HPV) is etiologically involved in driving these carcinomas .",
    "category": "etiology",
    "difficulty": "simple"
  },
  {
    "id": "Q171",
    "q": "How does human papillomavirus (HPV) infection status influence survival outcomes in oropharyngeal cancer?",
    "a": "HPV-related cases of oropharyngeal cancer have consistently better survival rates compared to non-HPV-related cases .",
    "category": "prognosis",
    "difficulty": "simple"
  },
  {
    "id": "Q172",
    "q": "What distinguishes malignant melanoma from sebaceous adenocarcinoma in the parotid gland on immunohistochemistry?",
    "a": "Malignant melanoma would typically express markers like S100 and SOX10, whereas sebaceous adenocarcinomas typically lack these but show sebaceous differentiation .",
    "category": "pathology",
    "difficulty": "complex"
  },
  {
    "id": "Q173",
    "q": "What is the typical presentation of early glottic (true vocal fold) laryngeal cancer?",
    "a": "Persistent hoarseness is the usual presenting symptom for cancers arising on the true vocal folds .",
    "category": "clinical_features",
    "difficulty": "simple"
  },
  {
    "id": "Q174",
    "q": "What is the primary treatment approach for Ewing's sarcoma of the pelvic bone?",
    "a": "Induction chemotherapy followed by an assessment to determine optimal local therapy, which usually involves radiotherapy or a combination of radiotherapy and delayed surgery to achieve complete resection .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q175",
    "q": "What are the essential diagnostic criteria for non-keratinizing squamous cell carcinoma of the sinonasal tract?",
    "a": "The essential criteria are an infiltrative poorly differentiated carcinoma with limited evidence of squamous differentiation .",
    "category": "diagnosis",
    "difficulty": "moderate"
  },
  {
    "id": "Q176",
    "q": "According to Indian national guidelines, what pathological findings mandate the use of post-mastectomy radiation in breast cancer?",
    "a": "Post-mastectomy radiation must be used in all patients with pathological tumors greater than 5 cm or if four or more axillary lymph nodes are positive .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q177",
    "q": "Which combination of immunotherapy drugs is the standard first-line treatment for intermediate- and poor-prognosis metastatic clear cell renal cell carcinoma?",
    "a": "The combination of nivolumab (anti-PD-1) and ipilimumab (anti-CTLA-4) has demonstrated significant improvement in overall and progression-free survival .",
    "category": "treatment",
    "difficulty": "complex"
  },
  {
    "id": "Q178",
    "q": "In urothelial carcinoma, what is a primary limitation for standard systemic chemotherapy?",
    "a": "Ineligibility to receive cisplatin affects approximately 50% of patients with metastatic urothelial carcinoma, limiting the use of the most effective standard front-line chemotherapy .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q179",
    "q": "In the liver tumor microenvironment, what factors drive immune escape in hepatocellular carcinoma?",
    "a": "Chronic inflammation and hypoxia induce the expression of immune checkpoint molecules (PD-L1, TIM-3) and recruit regulatory T cells and tumor-associated macrophages, facilitating immune escape .",
    "category": "mechanism",
    "difficulty": "complex"
  },
  {
    "id": "Q180",
    "q": "What is the treatment of choice for Stage I (T1N0M0) esophageal cancer?",
    "a": "Surgery (esophagectomy) is the treatment of choice. Radiation therapy is offered if the patient is medically unfit or unwilling for surgery .",
    "category": "treatment",
    "difficulty": "simple"
  },
  {
    "id": "Q181",
    "q": "What are the common chemotherapy agents used concurrently with radiation in the definitive treatment of unresectable esophageal tumors?",
    "a": "Common regimens use concurrent radiation with either cisplatin and 5-fluorouracil (5-FU), or paclitaxel/docetaxel with 5-FU .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q182",
    "q": "What organs are considered critical structures that limit the dose of radiotherapy for pelvic tumors like cervical cancer?",
    "a": "Critical organs for radiotherapy in the pelvic region include the loops of the small intestine, rectum, and bladder .",
    "category": "side_effects",
    "difficulty": "moderate"
  },
  {
    "id": "Q183",
    "q": "When is adjuvant radiotherapy indicated for well-differentiated thyroid cancer?",
    "a": "It is indicated in tumors with insufficient or no accumulation of radioactive iodine, or when the tumor infiltrates surrounding connective tissue .",
    "category": "treatment",
    "difficulty": "complex"
  },
  {
    "id": "Q184",
    "q": "How is an adequate margin typically defined in breast-conserving surgery?",
    "a": "Margins should be technically free of tumor. While guidelines variably define an adequate margin from 1 mm to 10 mm, obtaining a clean negative margin is universally desirable .",
    "category": "surgery",
    "difficulty": "moderate"
  },
  {
    "id": "Q185",
    "q": "What is the definition and goal of palliative care in oncology?",
    "a": "Palliative care controls symptoms, relieves emotional and physical suffering from cancer and its treatment, and improves quality of life regardless of the disease stage .",
    "category": "general",
    "difficulty": "simple"
  },
  {
    "id": "Q186",
    "q": "During a psycho-oncological assessment for a child with cancer, what is the primary goal of the investigative phase?",
    "a": "The goal is to identify available coping and adaptation strategies to create necessary support, rather than to uncover conflicts .",
    "category": "general",
    "difficulty": "moderate"
  },
  {
    "id": "Q187",
    "q": "What is the recommended systemic therapy for patients with Stage IVB (distant metastatic) cervical cancer?",
    "a": "Patients with Stage IVB cervical cancer are treated primarily with systemic platinum-based chemotherapy regimens, sometimes in combination with biologic therapy like bevacizumab .",
    "category": "treatment",
    "difficulty": "moderate"
  },
  {
    "id": "Q188",
    "q": "What is a cornerstone of management for pediatric patients diagnosed with familial polyposis syndromes?",
    "a": "Prophylactic surgery, education of families about pre-disposing syndromes, and the implementation of regular screening colonoscopies are essential .",
    "category": "treatment",
    "difficulty": "simple"
  },
  {
    "id": "Q189",
    "q": "What serum markers are monitored in pediatric ovarian germ cell tumors?",
    "a": "Alpha-fetoprotein (AFP), human chorionic gonadotropin (beta-hCG), and lactate dehydrogenase (LDH) are critical markers .",
    "category": "biomarker",
    "difficulty": "moderate"
  },
  {
    "id": "Q190",
    "q": "Pediatric adrenocortical tumors are frequently associated with which familial cancer predisposition syndrome?",
    "a": "They are frequently associated with Li-Fraumeni syndrome, caused by a germline pathogenic variant in the TP53 gene .",
    "category": "etiology",
    "difficulty": "moderate"
  },
  {
    "id": "Q191",
    "q": "How can clinicians monitor antigen-specific T cell activity in patients undergoing immunotherapy?",
    "a": "Clinicians can predict HLA-binding epitopes and use tetramer-based staining strategies to test peripheral blood or TILs for reactivity at different time points .",
    "category": "investigation",
    "difficulty": "complex"
  },
  {
    "id": "Q192",
    "q": "How does the gut microbiota influence systemic cancer immunotherapy?",
    "a": "The microbiome modulates the host's immune set point, and altering it through probiotics or carefully selected antibiotics may enhance the efficacy of treatments like PD-1 inhibitors .",
    "category": "mechanism",
    "difficulty": "complex"
  },
  {
    "id": "Q193",
    "q": "What direct impact does targeting the MAPK pathway (e.g., using BRAF inhibitors) have on the immune system in melanoma?",
    "a": "It increases melanoma antigen expression, CD8+ T cell infiltration, Class I MHC upregulation, and decreases immunosuppressive cytokine production .",
    "category": "mechanism",
    "difficulty": "complex"
  },
  {
    "id": "Q194",
    "q": "What are the recommended screening tests for prostate cancer in men over 50 years of age?",
    "a": "An annual digital rectal examination and a Prostate Specific Antigen (PSA) blood level determination are recommended .",
    "category": "investigation",
    "difficulty": "simple"
  },
  {
    "id": "Q195",
    "q": "How does the WHO classify tumors of the central nervous system (CNS) based on growth rate and aggressiveness?",
    "a": "They are classified by histopathological grade into low grade (slow-growing, types I and II) and high grade (rapidly growing and aggressive, types III and IV) .",
    "category": "pathology",
    "difficulty": "simple"
  },
  {
    "id": "Q196",
    "q": "When diagnosing operable breast cancer, why is the evaluation of ER, PR, and HER2/neu status essential?",
    "a": "These biological characteristics dictate systemic treatment options, such as prescribing endocrine therapy for ER/PR-positive tumors or targeted therapy like Trastuzumab for HER2/neu-amplified tumors .",
    "category": "biomarker",
    "difficulty": "moderate"
  },
  {
    "id": "Q197",
    "q": "Why is it important to assess serum tumor marker decline during chemotherapy for non-seminomatous germ cell tumors?",
    "a": "It allows for the early identification of therapeutic failure, providing critical information on the efficacy of the current treatment regimen .",
    "category": "biomarker",
    "difficulty": "complex"
  },
  {
    "id": "Q198",
    "q": "What is an advance directive in the context of end-of-life planning for cancer patients?",
    "a": "An advance directive is a legal document that states a patient's wishes in writing regarding their end-of-life medical care .",
    "category": "general",
    "difficulty": "simple"
  },
  {
    "id": "Q199",
    "q": "If a cancer patient is experiencing distress, what specific symptom might they ask their doctor to evaluate to determine if it is purely distress-related?",
    "a": "Patients are encouraged to ask their care team, \"Is my symptom(s) part of being distressed?\" to determine if physical symptoms are manifestations of psychological distress .",
    "category": "general",
    "difficulty": "moderate"
  },
  {
    "id": "Q200",
    "q": "What accelerated approval did the FDA grant for sacituzumab govitecan in April 2020?",
    "a": "The FDA granted accelerated approval to sacituzumab govitecan for adult patients with metastatic triple-negative breast cancer (TNBC) that has failed prior multi-line treatments .",
    "category": "treatment",
    "difficulty": "moderate"
  }
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def token_f1(pred, gt):
    pt = pred.lower().split(); gt_t = gt.lower().split()
    common = Counter(pt) & Counter(gt_t); n = sum(common.values())
    if n == 0: return 0.0, 0.0, 0.0
    p = n/len(pt); r = n/len(gt_t)
    return p, r, 2*p*r/(p+r)

def exact_match(pred, gt):
    pred_clean = pred.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    gt_clean = gt.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    return int(pred_clean == gt_clean)

def distinct_n(texts, n):
    ng = []
    for t in texts:
        toks = t.lower().split()
        ng.extend(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
    return len(set(ng))/len(ng) if ng else 0.0

def ndcg_at_k(relevances, k):
    rels = relevances[:k]
    dcg  = sum(r/math.log2(i+2) for i,r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum(r/math.log2(i+2) for i,r in enumerate(ideal))
    return dcg/idcg if idcg > 0 else 0.0

def retrieval_metrics(chunks, gt, k=5):
    """Use SBERT to proxy-label chunks as relevant if sim >= threshold."""
    gt_emb = sbert.encode(gt, convert_to_tensor=True)
    rels, scores = [], []
    for c in chunks[:k]:
        c_emb  = sbert.encode(c, convert_to_tensor=True)
        sim    = util.cos_sim(c_emb, gt_emb).item()
        scores.append(sim)
        rels.append(1 if sim >= RELEVANCE_THRESH else 0)

    n_rel      = sum(rels)
    precision  = n_rel / k
    recall     = n_rel / max(1, sum(rels))   # proxy: assume all relevant = retrieved relevant
    hit        = 1 if n_rel > 0 else 0
    mrr        = 0.0
    for rank, r in enumerate(rels, 1):
        if r: mrr = 1.0/rank; break
    ndcg       = ndcg_at_k(rels, k)
    avg_score  = float(np.mean(scores)) if scores else 0.0
    return precision, recall, hit, mrr, ndcg, avg_score

def context_relevance(question, chunks):
    """Avg SBERT sim between question and each chunk."""
    if not chunks: return 0.0
    q_emb = sbert.encode(question, convert_to_tensor=True)
    sims  = [util.cos_sim(sbert.encode(c, convert_to_tensor=True), q_emb).item() for c in chunks]
    return float(np.mean(sims))

def answer_relevance(question, answer):
    q_emb = sbert.encode(question, convert_to_tensor=True)
    a_emb = sbert.encode(answer,   convert_to_tensor=True)
    return util.cos_sim(a_emb, q_emb).item()

_llm_first_error_printed = False  # print the real error once for diagnosis

def _llm_call_with_retry(messages, max_tokens, retries=3):
    """Call the LLM with exponential backoff on failure."""
    global _llm_first_error_printed
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=messages,
                max_tokens=max_tokens, temperature=0.0
            )
            return resp
        except Exception as e:
            if not _llm_first_error_printed:
                print(f"  [LLM-WARN] API call failed (attempt {attempt+1}/{retries}): {e}")
                _llm_first_error_printed = True
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)  # 2s, 4s between retries
                time.sleep(wait)
    return None

def faithfulness_score(question, answer, context):
    prompt = (
        "You are a medical fact-checker.\n\n"
        f"Context:\n{context[:800]}\n\nAnswer:\n{answer}\n\nQuestion:\n{question}\n\n"
        "Does the answer contain claims NOT supported by the context? "
        'Respond ONLY with JSON: {"faithfulness": <0.0-1.0>, "reason": "<one sentence>"}\n'
        "1.0 = fully grounded, 0.0 = fully hallucinated."
    )
    resp = _llm_call_with_retry([{"role":"user","content":prompt}], max_tokens=120)
    if resp:
        m = re.search(r"\{.*?\}", resp.choices[0].message.content, re.DOTALL)
        if m:
            try: return float(json.loads(m.group()).get("faithfulness", 0.5))
            except Exception: pass
    return float("nan")

def llm_judge(question, answer, gt, context):
    prompt = (
        "You are an expert oncology evaluator. Rate the generated answer 1-10.\n\n"
        f"Question: {question}\nReference: {gt}\nGenerated: {answer}\n\n"
        'Respond ONLY with JSON: {"score": <int 1-10>, "reason": "<one sentence>"}'
    )
    resp = _llm_call_with_retry([{"role":"user","content":prompt}], max_tokens=150)
    if resp:
        m = re.search(r"\{.*?\}", resp.choices[0].message.content, re.DOTALL)
        if m:
            try: return float(json.loads(m.group()).get("score", 0)) / 10.0
            except Exception: pass
    return float("nan")

def scope_judge(question, answer, gt, context):
    """Score on 1-5 scale per dimension; weighted/average also on 1-5."""
    prompt = (
        "You are an expert oncology evaluator. Score the answer strictly (1-5 each):\n"
        "S-Sufficiency: does the answer cover all key facts?\n"
        "C-Correctness: is every claim factually accurate?\n"
        "O-Organization: is it clearly structured?\n"
        "P-Pertinence: does it directly address the question?\n"
        "E-Exactness: does it match the reference answer precisely?\n\n"
        f"Question: {question}\nContext: {context[:500]}\nAnswer: {answer}\nReference: {gt}\n\n"
        'Respond ONLY with JSON: {"S":<1-5>,"C":<1-5>,"O":<1-5>,"P":<1-5>,"E":<1-5>}'
    )
    resp = _llm_call_with_retry([{"role":"user","content":prompt}], max_tokens=100)
    if resp:
        m = re.search(r"\{.*?\}", resp.choices[0].message.content, re.DOTALL)
        if m:
            try:
                d  = json.loads(m.group())
                # Keep raw 1-5 values
                sc = {k: float(min(max(d.get(k, 3), 1), 5)) for k in "SCOPE"}
                sc["weighted"] = sum(SCOPE_WEIGHTS[k] * sc[k] for k in "SCOPE") * 5
                sc["average"]  = float(np.mean([sc[k] for k in "SCOPE"]))
                return sc
            except Exception: pass
    return {k: float("nan") for k in list("SCOPE")+["weighted","average"]}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  --  Run RAG pipeline
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  ONCOLOGY RAG -- ADVANCED EVALUATION  (Q001-Q200)")
print("="*72 + "\n")

(questions, ground_truths, ids, categories, difficulties,
 answers, contexts, rerank_scores_all, iterations_all) = (
    [], [], [], [], [], [], [], [], []
)

for item in EVAL_QA:  # Q001–Q200
    q = item["q"]
    print(f"  [{item['id']}] {q[:68]}...")

    # Retrieve then rerank (with scores)
    raw_results = retrieve(q, top_k=10)
    iteration   = 1

    top5, top5_scores = rerank_with_scores(q, raw_results, top_k=5)

    ctx    = [r.metadata["text"] for r in top5]
    answer = generate_answer(q, "\n".join(ctx))

    questions.append(q); ground_truths.append(item["a"])
    ids.append(item["id"]); categories.append(item["category"])
    difficulties.append(item["difficulty"])
    answers.append(answer); contexts.append(ctx)
    rerank_scores_all.append(top5_scores)
    iterations_all.append(iteration)

print("\n[OK] Pipeline complete. Computing metrics...\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  --  Per-question metrics
# ─────────────────────────────────────────────────────────────────────────────
(bleu1_s, bleu4_s, gleu_s, meteor_s,
 rouge1_s, rouge2_s, rougel_s, rougeLsum_s,
 prec_s, rec_s, f1_s, em_s,
 sbert_s, judge_s, faith_s,
 ctx_rel_s, ans_rel_s,
 prec5_s, rec5_s, hit5_s, mrr_s, ndcg5_s, avg_rk_s,
 scope_s) = ([] for _ in range(24))

for ans, gt, ctx, q, rk_scores in zip(answers, ground_truths, contexts, questions, rerank_scores_all):
    ref_t  = gt.split(); pred_t = ans.split()
    ctx_str = "\n".join(ctx)

    bleu1_s.append(sentence_bleu([ref_t], pred_t, weights=(1,0,0,0), smoothing_function=smoother))
    bleu4_s.append(sentence_bleu([ref_t], pred_t, smoothing_function=smoother))
    gleu_s.append(sentence_gleu([ref_t], pred_t))
    meteor_s.append(nltk_meteor([ref_t], pred_t))

    r = rouge.score(gt, ans)
    rouge1_s.append(r["rouge1"].fmeasure)
    rouge2_s.append(r["rouge2"].fmeasure)
    rougel_s.append(r["rougeL"].fmeasure)
    rougeLsum_s.append(r["rougeLsum"].fmeasure)

    p, rec, f1 = token_f1(ans, gt)
    prec_s.append(p); rec_s.append(rec); f1_s.append(f1)
    em_s.append(exact_match(ans, gt))

    sbert_s.append(util.cos_sim(
        sbert.encode(ans, convert_to_tensor=True),
        sbert.encode(gt,  convert_to_tensor=True)
    ).item())

    pr5, re5, h5, mrr, ndcg5, avg_rk = retrieval_metrics(ctx, gt)
    prec5_s.append(pr5); rec5_s.append(re5); hit5_s.append(h5)
    mrr_s.append(mrr); ndcg5_s.append(ndcg5); avg_rk_s.append(avg_rk)

    ctx_rel_s.append(context_relevance(q, ctx))
    ans_rel_s.append(answer_relevance(q, ans))
    faith_s.append(faithfulness_score(q, ans, ctx_str))
    time.sleep(0.4)   # brief pause to avoid rate-limit burst
    judge_s.append(llm_judge(q, ans, gt, ctx_str))
    time.sleep(0.4)
    scope_s.append(scope_judge(q, ans, gt, ctx_str))
    time.sleep(0.4)

d1 = distinct_n(answers, 1)
d2 = distinct_n(answers, 2)
avg_iters = float(np.mean(iterations_all))
# Sigmoid-normalize rerank logits to 0-1 before averaging
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
norm_scores = [float(np.mean(_sigmoid(np.array(s)))) for s in rerank_scores_all if len(s) > 0]
avg_conf = float(np.nanmean(norm_scores)) if norm_scores else float("nan")

print("[..] Computing BERTScore...\n")
_, _, F1_b = bertscore(answers, ground_truths, lang="en")
avg_bert_f1 = F1_b.mean().item()

scope_agg = {k: float(np.nanmean([s[k] for s in scope_s]))
             for k in list("SCOPE") + ["weighted", "average"]}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  --  Report
# ─────────────────────────────────────────────────────────────────────────────
report = f"""
{'='*72}
ONCOLOGY RAG -- ADVANCED EVALUATION REPORT  (Q001-Q200)
{'='*72}
Timestamp           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Questions evaluated : {len(questions)}

-- Accuracy & F1 {'-'*53}
Exact Match (EM)        : {np.mean(em_s):.4f}
Token Precision         : {np.mean(prec_s):.4f}
Token Recall            : {np.mean(rec_s):.4f}
Token F1                : {np.mean(f1_s):.4f}

-- BLEU / GLEU / METEOR {'-'*46}
BLEU-1                  : {np.mean(bleu1_s):.4f}
BLEU-4                  : {np.mean(bleu4_s):.4f}
GLEU                    : {np.mean(gleu_s):.4f}
METEOR                  : {np.mean(meteor_s):.4f}

-- ROUGE {'-'*62}
ROUGE-1                 : {np.mean(rouge1_s):.4f}
ROUGE-2                 : {np.mean(rouge2_s):.4f}
ROUGE-L                 : {np.mean(rougel_s):.4f}
ROUGE-Lsum              : {np.mean(rougeLsum_s):.4f}

-- DISTINCT (Diversity) {'-'*47}
DISTINCT-1              : {d1:.4f}
DISTINCT-2              : {d2:.4f}

-- Semantic Similarity {'-'*49}
SBERT Cosine Sim        : {np.mean(sbert_s):.4f}
BERTScore F1            : {avg_bert_f1:.4f}
Answer Relevance        : {np.mean(ans_rel_s):.4f}

-- Retrieval Quality (proxy-labelled @ thresh={RELEVANCE_THRESH}) {'-'*6}
Precision@5             : {np.mean(prec5_s):.4f}
Recall@5                : {np.mean(rec5_s):.4f}
Hit-Rate@5              : {np.mean(hit5_s):.4f}
MRR                     : {np.mean(mrr_s):.4f}
NDCG@5                  : {np.mean(ndcg5_s):.4f}
Avg Rerank Score        : {np.mean(avg_rk_s):.4f}
Context Relevance       : {np.mean(ctx_rel_s):.4f}

-- Faithfulness / Hallucination {'-'*39}
Faithfulness (LLM)      : {float(np.nanmean(faith_s)):.4f}

-- Agentic Metrics {'-'*52}
Avg Agent Iterations    : {avg_iters:.2f}
Avg Confidence Score    : {avg_conf:.4f}

-- S.C.O.P.E Framework (1-5 scale) {'-'*36}
Sufficiency    (S x0.20): {scope_agg['S']:.2f} / 5
Correctness    (C x0.30): {scope_agg['C']:.2f} / 5
Organization   (O x0.15): {scope_agg['O']:.2f} / 5
Pertinence     (P x0.25): {scope_agg['P']:.2f} / 5
Exactness      (E x0.10): {scope_agg['E']:.2f} / 5
SCOPE Weighted Avg      : {scope_agg['weighted']:.2f} / 5
SCOPE Simple Avg        : {scope_agg['average']:.2f} / 5

-- LLM-as-a-Judge (0-1) {'-'*47}
LLM Judge Score         : {float(np.nanmean(judge_s)):.4f}

{'='*72}
"""

print(report)

# per-question table
print("-- Per-question breakdown " + "-"*47)
hdr = f"{'ID':<6} {'Cat':<12} {'Diff':<8} {'F1':>5} {'R-1':>5} {'SBERT':>6} {'Faith':>6} {'Judge':>6} {'NDCG':>6}"
print(hdr); print("-"*68)
for i in range(len(questions)):
    def _f(v): return f"{v:.3f}" if not (isinstance(v,float) and math.isnan(v)) else "  N/A"
    print(f"{ids[i]:<6} {categories[i]:<12} {difficulties[i]:<8} "
          f"{f1_s[i]:>5.3f} {rouge1_s[i]:>5.3f} {sbert_s[i]:>6.3f} "
          f"{_f(faith_s[i]):>6} {_f(judge_s[i]):>6} {ndcg5_s[i]:>6.3f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  --  Save
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("evaluation", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

with open(f"evaluation/eval_report_{ts}.txt", "w", encoding="utf-8") as f:
    f.write(report)
print(f"[OK] Report -> evaluation/eval_report_{ts}.txt")

out_df = pd.DataFrame({
    "id": ids, "category": categories, "difficulty": difficulties,
    "question": questions, "generated_answer": answers, "ground_truth": ground_truths,
    "exact_match": em_s, "token_f1": f1_s,
    "bleu1": bleu1_s, "bleu4": bleu4_s, "gleu": gleu_s, "meteor": meteor_s,
    "rouge1": rouge1_s, "rouge2": rouge2_s, "rougeL": rougel_s, "rougeLsum": rougeLsum_s,
    "sbert": sbert_s, "bert_f1": F1_b.tolist(),
    "answer_relevance": ans_rel_s, "context_relevance": ctx_rel_s,
    "faithfulness": faith_s, "llm_judge": judge_s,
    "precision_at5": prec5_s, "recall_at5": rec5_s, "hit_rate_at5": hit5_s,
    "mrr": mrr_s, "ndcg_at5": ndcg5_s, "avg_rerank_score": avg_rk_s,
    "agent_iterations": iterations_all,
    "scope_S": [s["S"] for s in scope_s], "scope_C": [s["C"] for s in scope_s],
    "scope_O": [s["O"] for s in scope_s], "scope_P": [s["P"] for s in scope_s],
    "scope_E": [s["E"] for s in scope_s],
    "scope_weighted": [s["weighted"] for s in scope_s],
    "scope_avg": [s["average"] for s in scope_s],
})

ep = f"evaluation/eval_results_{ts}.xlsx"
out_df.to_excel(ep, index=False)
print(f"[OK] Excel  -> {ep}\n")
print("[DONE] Evaluation complete!")
