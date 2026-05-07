def verify_answer(answer, context):
    # simple grounding check
    for word in answer.split():
        if word.lower() in context.lower():
            return True

    return False
