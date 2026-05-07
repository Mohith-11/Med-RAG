from vectorstore.query_rewrite.rewrite import rewrite_query
from retrieval.reasoning import decompose_query
from retrieval.crag import crag_retrieve_multi
from retrieval.filter import filter_metadata
from retrieval.rerank import rerank
from retrieval.compress import compress_context
from generator.generate import generate_answer
from generator.verify import verify_answer

# 🔥 User query
query = input("Enter your medical question: ")

# 🔥 Step 1: Rewrite query
query = rewrite_query(query)

# 🔥 Step 2: Multi-step reasoning
sub_queries = decompose_query(query)


# 🔥 Step 3: CRAG retrieval
results = crag_retrieve_multi(sub_queries)[:10]

# 🔥 limit noisy results
results = results[:10]

# 🔥 Step 4: Metadata filtering
results = filter_metadata(results)

# 🔥 Step 5: Rerank
results = rerank(query, results, top_k=5)

# 🔥 Step 6: Context compression WITH citations
compressed_context = compress_context(results)

# 🔥 Step 7: Generate answer
answer = generate_answer(query, compressed_context)



# 🔥 Final answer
print("\n🧠 Answer:\n")
print(answer)

# 🔥 Sources
print("\n📚 Sources:\n")

for i, r in enumerate(results):
    print(f"[{i+1}] {r.metadata['text'][:250]}...\n")