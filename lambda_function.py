import json
import time
import boto3
import numpy as np
from openai import OpenAI
import os

s3vectors_client = boto3.client('s3vectors')
athena_client = boto3.client('athena')

def query_athena(query, params):
    # Start the Athena query execution
    start_query_response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': os.environ['ATHENA_DATABASE']
        },
        ResultConfiguration={
            'OutputLocation': os.environ['ATHENA_OUTPUT_S3']
        },
        ExecutionParameters=params
    )
    print("Query execution started:", start_query_response)

    query_execution_id = start_query_response['QueryExecutionId']
    
    # Poll the query status until it completes
    while True:
        status_response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        state = status_response['QueryExecution']['Status']['State']
        reason = status_response['QueryExecution']
        
        if state == 'SUCCEEDED':
            break
        elif state in ['FAILED', 'CANCELLED']:
            raise Exception(f"Query {state}: {reason}")
        
        time.sleep(0.2)  # Poll every 0.2 seconds
        
    results_response = athena_client.get_query_results(QueryExecutionId=query_execution_id)
    
    if not results_response or 'ResultSet' not in results_response or 'Rows' not in results_response['ResultSet']:
        return []
 
    def unpack_athena_row(row):
        return [obj['VarCharValue'] for obj in row['Data']]
 
    # Unpack the results into a list of dictionaries, using the header row as keys
    header, *rows = results_response['ResultSet']['Rows']
    unpacked_header = unpack_athena_row(header)
    unpacked_results = [dict(zip(unpacked_header, unpack_athena_row(row))) for row in rows]    
    return unpacked_results

def get_course_data(course_list, school_name):
    school_name_code_lookup = {
        "university of wyoming": "UWYO",
    }
    school_code = school_name_code_lookup.get(school_name.lower())    
    # Build the SQL query to fetch course data based on course titles and codes
    query = query = f"""
    SELECT id, data_title, data_code, data_desc, dse_skills
    FROM courses
    WHERE data_src = '{school_code}'
        AND data_code IN ({', '.join(['?']*len(course_list))})
    """
    
    return query_athena(query, [code for _, code in course_list])

def get_job_data(job_ids):
    query = f"""
    SELECT *
    FROM jobs
    WHERE id IN ({', '.join(['?']*len(job_ids))})
    """
    return query_athena(query, job_ids)

def load_embeddings(index_arn, vector_keys):
    try:
        vectors = s3vectors_client.get_vectors(
            indexArn=index_arn,
            keys=vector_keys,
            returnData=True
        )
    except Exception as e:
        raise Exception(f"Error retrieving embeddings from S3Vectors: {str(e)}")
    
    if vectors['ResponseMetadata']['HTTPStatusCode'] != 200:
        raise Exception(f"Failed to retrieve embeddings from S3Vectors: {vectors['ResponseMetadata']['HTTPStatusCode']}")
    
    return vectors['vectors']

def matching_skills(student_skills, student_skill_groups, job_skills, job_skill_groups):
    common_skills = list(set(student_skills) & set(job_skills))
    common_skill_groups = list(set(student_skill_groups.keys()) & set(job_skill_groups.keys()))
    
    return common_skills, common_skill_groups

### OPENAI API RELATED ###

def init_client():
    # Get OpenAI key from aws secrets manager and return OpenAI client
    secrets_client = boto3.client('secretsmanager')
    secret_response = secrets_client.get_secret_value(SecretId=os.environ['OPENAI_API_KEY_SECRET'])
    secret_string = secret_response['SecretString']
    api_key = json.loads(secret_string).get('OPENAI_API_KEY')
    return OpenAI(api_key=api_key)

def chatgpt_send_messages_json(messages, json_schema_wrapper, model, client, service_tier="standard"):
    json_response = client.chat.completions.create(
        model=model,
        messages=messages,
        #service_tier=service_tier,  # "priority", "standard", "flex" // Priority only works for gpt-5 and its mini version.
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": json_schema_wrapper["name"],
                "strict": True,
                "schema": json_schema_wrapper["schema"]
            }
        },
        temperature=1 if "5" in model else 0.2,
        top_p=1.0
    )
    json_response_content = json_response.choices[0].message.content
    return json.loads(json_response_content)

def normalize_string_list(items):
    if not items:
        return []
    # strip, dedupe (preserve order), drop empties
    cleaned = [s.strip() for s in items if isinstance(s, str)]
    seen, out = set(), []
    for s in cleaned:
        low = s.lower()
        if s and low not in seen:
            seen.add(low)
            out.append(s)
    return out

def build_final_filter_payload(top_jobs_data, com_skills, com_skill_groups, summary_text):
    jobs = []
    for j in top_jobs_data or []:
        ja = j.get("job_analysis", {}) or {}
        jobs.append({
            "id": j.get("id", "") or "",
            "title": j.get("title", "") or "",
            "job_analysis": {
                "summary": ja.get("summary", "") or "",
                "experience_required": normalize_string_list(ja.get("experience_required")),
                "experience_preferred": normalize_string_list(ja.get("experience_preferred")),
                "software_required": normalize_string_list(ja.get("software_required")),
                "software_preferred": normalize_string_list(ja.get("software_preferred")),
                "certifications": normalize_string_list(ja.get("certifications")),
                "education": normalize_string_list(ja.get("education")),
                "musthaves": normalize_string_list(ja.get("musthaves")),
                "optional_recommended": normalize_string_list(ja.get("optional_recommended")),
                "at_glance": normalize_string_list(ja.get("at_glance")),
                "expertise_ranking": (ja.get("expertise_ranking") or "").strip()
            },
            "skills": normalize_string_list(j.get("skills")),
            "skill_groups": j.get("skill_groups", {}) or {}
        })

    payload = {
        "schema_version": "pre-llm-input/v2",
        "student": {
            "summary": (summary_text or "").strip(),
            "skills": normalize_string_list(com_skills or []),
            "skill_groups": normalize_string_list(com_skill_groups or [])
        },
        "jobs": jobs
    }
    return payload

def get_prompt_plus_schema(top_jobs_data, com_skills, com_skill_groups, summary_text):
    _SYSTEM_PROMPT = """
        You are the “Final Filtering & AI Check” for a job-matching pipeline.

        VOICE & STYLE
        - Write directly to the candidate in second person. Always use “you/your”.
        - Never use third-person phrasing such as “the candidate’s ability” or “their skills”.

        ASSUMPTION ABOUT THE CANDIDATE
        - Assume the candidate is a current university student (bachelor’s or master’s).
        - Do not invent professional full-time years of experience.

        TASK
        For each job in the provided JSON payload, return three fields per job:
        1) justification — exactly ONE sentence (second person) tying your background to a CORE requirement of the job.
           - Example pattern: “Your experience in X aligns with the job’s Y requirement.”
           - ≤ 200 characters, plain text, no emojis, no markdown.
        2) next_steps — TWO to THREE short sentences (second person) telling what to check/do.
           - Use directive verbs: confirm, highlight, prepare, obtain, practice, document.
           - Be concise and professional; no promises or sensitive inferences.
           - If the job appears borderline for a student (see BORDERLINE RULES), include one sentence that acknowledges this and advises how to position your profile (internships, projects, coursework, assistant/associate titles).
        3) compatibility_score_10 — a NUMBER from 0.0 to 10.0 with ONE decimal place, computed by the rubric below.

        JUSTIFICATION SELECTION RULES
        - X must be a single student item from the input (a skill, a term from student.summary, or a skill_group label).
        - Y must be ONE exact string taken from one of:
          - job_analysis.musthaves | job_analysis.software_required | job_analysis.experience_required | job_analysis.certifications | job_analysis.education
          - OR job.skills (not skill_groups).
        - Do NOT use category labels like “musthaves” or “software_required” in the sentence; use the verbatim requirement string (e.g., “Microsoft Excel”, “Criminal background check with fingerprinting”).

        NEXT STEPS RULES
        - Mention at least one exact requirement string from the job in the first sentence when any of {job_analysis.musthaves, job_analysis.software_required} is non-empty.
        - Do not mention certifications unless job_analysis.certifications.length > 0.
        - Do not reference fields that are empty in the job payload.
        - If any of these phrases appear in job_analysis.musthaves, you must surface them explicitly: “background check”, “E-Verify”, “valid driver’s license”, “ability to travel”.

        BORDERLINE RULES (for students)
        - Treat a job as borderline if ANY of the following are true:
          1) job_analysis.expertise_ranking is "intermediate" or "senior"
          2) job.title contains “Senior”, “Principal”, or “Lead” (case-insensitive)
          3) job_analysis.experience_required contains any number ≥ 3 (e.g., “3-4 years”, “5+ years”)
        - When borderline, add ONE concise sentence in next_steps such as:
          “This role typically expects 3+ years; emphasize internships, major projects, and relevant coursework to bridge the gap.”

        STANDARDIZED SCORING RUBRIC (/10, deterministic; ONE decimal)
        - Use ONLY student.skills and student.skill_groups (do not extract new skills from summary) and the job’s provided arrays. Case-insensitive exact match on full strings.
        - Define:
          • JS = set(job.skills)
          • SW_req = job_analysis.software_required
          • MH = job_analysis.musthaves
          • GJ = set(keys of job.skill_groups)
          • SS = set(student.skills)
          • SG = set(student.skill_groups)
        - Compute:
          1) skill_coverage = |{s ∈ JS : s matches any in SS}| / max(1, |JS|)
          2) software_coverage = |{s ∈ SW_req : s matches any in SS}| / max(1, |SW_req|); if SW_req empty → 1.0
          3) group_alignment = |{g ∈ GJ : g matches any in SG}| / max(1, |GJ|); if GJ empty → 1.0
          4) musthave_penalty_count = number of items in MH that are NOT matched by SS and are NOT administrative. “Administrative” means the string contains (case-insensitive): “background check”, “e-verify”, “driver’s license”, “ability to travel”. Cap at 2.
          5) composite = 0.6*skill_coverage + 0.3*software_coverage + 0.1*group_alignment
          6) composite_adj = composite * (1 - 0.3*musthave_penalty_count)

          # Student-aware experience barrier (applies to all candidates as they are students)
          7) experience_years_flag:
             - Inspect job_analysis.experience_required strings for integers. Let Y = max integer found (or 0 if none).
             - If Y >= 5 → exp_penalty = 0.20
               Else if Y >= 3 → exp_penalty = 0.10
               Else → exp_penalty = 0.00
          8) senior_title_flag:
             - If job.title contains “Senior”, “Principal”, or “Lead” (case-insensitive) → title_penalty = 0.05 else 0.00

          9) expertise_adjustment (from ranking):
             - if job_analysis.expertise_ranking == "intermediate" → −0.05
             - if "senior" → −0.10
             - else 0.00

          10) education_bonus (student assumption):
             - if job_analysis.education contains “Bachelor” or “Master” (case-insensitive) → +0.02 else +0.00

        - Final score:
          • composite_exp = composite_adj * (1 - exp_penalty) * (1 - title_penalty)
          • raw = composite_exp + expertise_adjustment + education_bonus
          • clamp raw to [0,1]
          • compatibility_score_10 = round-half-up(10 * raw, 1)   # ONE decimal place
          • clamp to [0.0, 10.0]
        - Output the score as a JSON NUMBER (not a string). It must be a multiple of 0.1 (e.g., 7.3, 8.0).

        WARNINGS POPULATION
        - If experience_years_flag > 0 or senior_title_flag > 0, add a warning entry of the form:
          "<job_id>: posting indicates seniority/≥3 years; treat as borderline for students."
        - If a job has no actionable fields (none of: musthaves, software_required, experience_required, certifications, education, skills), add:
          "<job_id>: limited structured requirements; guidance is generic."

        GROUNDING & DATA USE
        - Use ONLY the data in the input payload; do not add external knowledge.
        - If a field is missing, simply omit it; do not guess.
        - Maintain the input order of jobs.

        OUTPUT FORMAT (STRICT)
        - Return ONLY a JSON object that validates against the provided JSON Schema.
        - Always include these top-level keys: schema_version, jobs, warnings, errors.
        - The warnings and errors fields must be arrays (they may be empty).
        - No extra keys, no markdown, no explanations.

        SAFETY
        - No personal attributes or demographic inferences. No medical/legal claims. Neutral tone.

        You will receive a single JSON object as the user message.
        Return ONLY a JSON object that validates against the provided JSON Schema.
    """

    _JSON_SCHEMA_WRAPPER = {
        "name": "final_filter_v2_response",
        "schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema#",
            "title": "Final Filtering & AI Check Output v2",
            "type": "object",
            "additionalProperties": False,
            "required": ["schema_version", "jobs", "warnings", "errors"],
            "properties": {
                "schema_version": {"type": "string", "const": "final-filter/v2"},
                "jobs": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "title", "justification", "next_steps", "compatibility_score_10"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "title": {"type": "string", "minLength": 1},
                            "justification": {"type": "string", "minLength": 10, "maxLength": 200},
                            "next_steps": {"type": "string", "minLength": 30, "maxLength": 480},
                            "compatibility_score_10": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 10,
                                "multipleOf": 0.1,
                                "description": "Deterministic score per rubric; number with one decimal (multiple of 0.1) in range 0.0–10.0."
                            }
                        }
                    }
                },
                "warnings": {"type": "array", "items": {"type": "string"}},
                "errors": {"type": "array", "items": {"type": "string"}}
            }
        }
    }

    payload = build_final_filter_payload(
        top_jobs_data=top_jobs_data,
        com_skills=com_skills,
        com_skill_groups=com_skill_groups,
        summary_text=summary_text
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]
    return messages, _JSON_SCHEMA_WRAPPER

def get_similar_jobs(course_ids):
    # Load vector embeddings from course_ids
    course_embeddings = load_embeddings(os.environ['COURSE_VECTORS_INDEX_ARN'], course_ids)
    print("Course embeddings:", len(course_embeddings))

    # Calculate the average vector for the course embeddings
    course_mean_vector = np.mean([
        np.array(vec['data']['float32'], dtype=float) for vec in course_embeddings
        ], axis=0).astype(np.float32).tolist()
 
    # Query top k jobs based on the average course embedding
    top_k_jobs = s3vectors_client.query_vectors(
        indexArn=os.environ['JOB_VECTORS_INDEX_ARN'],
        topK=10,
        queryVector= {
            "float32": course_mean_vector
        },
        returnDistance=True
    )
    if top_k_jobs['ResponseMetadata']['HTTPStatusCode'] != 200:
        raise Exception(f"Failed to query embeddings from S3Vectors: {top_k_jobs['ResponseMetadata']['HTTPStatusCode']}")

    print("Top K jobs retrieved:", top_k_jobs["vectors"])
    return top_k_jobs["vectors"]

from time import perf_counter
def _timeit(f):
    def wrap(*a, **kw):
        t=perf_counter(); r=f(*a, **kw)
        print(f"{f.__name__} took {(perf_counter()-t)*1000:.3f} ms")
        return r
    return wrap

@_timeit
def lambda_handler(event, context):

    # Input validation, check if body is present
    if "body" not in event or not event["body"]:
        return {
            'status': 400,
            'body': 'Missing body in request'
        }
    elif type(event["body"]) is str:
        body = json.loads(event["body"])
    else:
        body = event["body"]

    print("Course load summary:")
    print("Input courses:", len(body["coursesList"]))
    print("Input source: ", body["source"])

    # Get course data from backend database, including ids
    course_data = get_course_data(body["coursesList"], body["source"])
    course_ids = [course["id"] for course in course_data]
    print(f"Found {len(course_data)}/{len(body['coursesList'])} courses")
    print(f"Course Ids: {course_ids}")

    # Find the top job matches given course ids using a vector embedding database
    similar_job_vectors = get_similar_jobs(course_ids)
    similar_job_ids = [job["key"] for job in similar_job_vectors]
    similar_job_data = get_job_data(similar_job_ids)

    # Print the top job ids in the dataset, as well as their distances
    print("Top job IDs and distances after skills parse:")
    for job in similar_jobs:
        print(f"Job ID: {job['id']}, Distance: {job['distance']}")

    model = "gpt-4.1-nano"
    first_com_skills, first_com_skill_groups = matching_skills(
        body["student_skill_list"],
        body["student_skill_groups"],
        similar_jobs[0]["skills"],
        similar_jobs[0]["skill_groups"]
    )

    for i, top_job_data in enumerate(similar_jobs):
        com_skills, com_skill_groups = matching_skills(
            body["student_skill_list"],
            body["student_skill_groups"],
            similar_jobs[i]["skills"],
            similar_jobs[i]["skill_groups"]
        )
        top_job_data["common_skills"] = com_skills
        top_job_data["common_skill_groups"] = com_skill_groups

    messages, json_schema_wrapper = get_prompt_plus_schema(
        top_jobs_data=similar_jobs,
        com_skills=first_com_skills,
        com_skill_groups=first_com_skill_groups,
        summary_text=body["summary"]
    )

    llm_result = chatgpt_send_messages_json(messages, json_schema_wrapper, model, client=init_client())["jobs"]

    jobs = llm_result["jobs"] if isinstance(llm_result, dict) and "jobs" in llm_result else llm_result
    jobs_sorted = sorted(jobs, key=lambda j: j["compatibility_score_10"], reverse=True)

    if isinstance(llm_result, dict):
        llm_result["jobs"] = jobs_sorted
    else:
        llm_result = jobs_sorted

    for job in llm_result:
        for data_job in similar_jobs:
            if data_job["id"] == job["id"]:
                job["url"] = data_job["url"]
                job["job_analysis"] = {
                        k: v for k, v in data_job["job_analysis"].items()
                        if k != "expertise_ranking_justification"
                    }
                job["skills"] = data_job["skills"]
                job["skill_groups"] = data_job["skill_groups"]
                job["common_skills"] = data_job["common_skills"]
                job["common_skill_groups"] = data_job["common_skill_groups"]
    
    highlight = "\n".join([
        f"{job_match["title"]}\n{job_match["justification"]}\nCompatibility: {job_match["compatibility_score_10"]}\n"
        for job_match in llm_result[:3]]
    )


    return {
        'status': 200,
        'body': {
            "job_matches": llm_result,
            "highlight": highlight
        }
    }
