import os
from flask import Flask, request, jsonify, send_from_directory

# Import your existing RAG pipeline
from vectorstore.query_rewrite.rewrite import rewrite_query
from retrieval.reasoning import decompose_query
from retrieval.crag import crag_retrieve_multi
from retrieval.filter import filter_metadata
from retrieval.rerank import rerank
from retrieval.compress import compress_context
from generator.generate import generate_answer

app = Flask(__name__, static_folder="frontend")

# Serve the frontend files
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

# Core Chat Endpoint
@app.route('/api/agent', methods=['POST'])
def api_agent():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing query in request body'}), 400
        
    original_query = data['query']
    
    try:
        # Run your pipeline
        query = rewrite_query(original_query)
        sub_queries = decompose_query(query)
        results = crag_retrieve_multi(sub_queries)[:10]
        results = filter_metadata(results)
        results = rerank(query, results, top_k=5)
        compressed_context = compress_context(results)
        answer = generate_answer(query, compressed_context)
        
        # Format sources for the frontend
        sources = []
        for r in results:
            page = r.metadata.get('page', 'N/A')
            source_doc = r.metadata.get('source', 'Unknown Document')
            
            # Extract just the filename if it's a full path
            if '/' in source_doc or '\\' in source_doc:
                source_doc = os.path.basename(source_doc)
                
            sources.append({
                "title": source_doc,
                "text": r.metadata.get('text', '')[:200] + "...",  # Preview chunk
                "page": page
            })
            
        return jsonify({
            "answer": answer,
            "sources": sources,
            "confidence": 0.95
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Optional: Evaluation Summary Mock Endpoint
@app.route('/api/evaluation/summary', methods=['GET'])
def eval_summary():
    # You can later hook this up to your full_eval.py / evaluate_200q.py data
    return jsonify({
        "questions_evaluated": 20,
        "avg_confidence": 0.824,
        "retrieval_quality": { "mrr": 0.8639, "hit_rate_at_5": 0.9750 },
        "generation_lexical": { "answer_f1": 0.4509, "rougeL": 0.2267 },
        "faithfulness_relevance": { "faithfulness": 0.8610 },
        "scope": { "precision": 3.92, "completeness": 3.96 }
    })

# Optional: Evaluation Results Mock Endpoint
@app.route('/api/evaluation/results', methods=['GET'])
def eval_results():
    return jsonify([
        {
            "question_id": "Q001",
            "question": "What are the three main anatomical divisions of the larynx?",
            "category": "diagnosis",
            "faithfulness": 0.95,
            "token_f1": 0.88,
            "judge_score": 4.5,
            "generated_answer": "The larynx is anatomically divided into the supraglottic larynx, the glottis, and the subglottis.",
            "ground_truth": "The larynx is anatomically divided into the supraglottic larynx, the glottis, and the subglottis."
        }
    ])

if __name__ == '__main__':
    print("🚀 Starting MedSpace AI Server on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)
