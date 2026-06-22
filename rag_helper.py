INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
""".strip()

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:
    def __init__(
        self,
        index,      
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='gemini-3.5-flash',
       
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, question):
        return self.index.search(
            question,
            num_results=5
        )


    def build_context(self, search_results):
        lines = []
        
        print(search_results)
        for doc in search_results:
            lines.append('content: ' + doc['content'])
            lines.append('filename: ' + doc['filename'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, question, search_results):
        context = self.build_context(search_results)
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            context=context
        )
        return prompt.strip()

    def llm(self, instructions, user_prompt, model='gemini-3.5-flash'):
        message_history = [
            {'role': 'developer', 'content': instructions},
            {'role': 'user', 'content': user_prompt}
        ]

        response = self.llm_client.chat.completions.create(
            model=model,
            messages=message_history
        )
        return response

    def rag(self, query, model='gemini-3.5-flash'):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(INSTRUCTIONS, prompt, model=model)
        return answer