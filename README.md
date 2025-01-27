# MetaphorRAG
👏🚶‍♂️👏🏰

## Description

A sample of a psychological RAG (Retrieval Augmented Generation) to help people dealing with mental problems.

The RAG system behaves like a CBT specialist. You should ask him a question in English about what is bothering you, and he will understandingly answer you using CBT methods.

Pipeline:

1. Get relevant problems from problem-solution datasets based on [CACTUS](https://huggingface.co/datasets/DLI-Lab/cactus), [MHCC](https://www.kaggle.com/datasets/melissamonfared/mental-health-counseling-conversations-k). 
2. Add CBT Principles.
3. Adapt relevant soutions based on user request.  

## Validation

100 questions were generated, and then they were asked to RAG pipeline.  After that, the relevance of the answers was evaluated with the help of an LLM judge (ChatGPT 4o Mini) using prompt:

> I'll send you some texts now. Your task is to evaluate from 1 to 5 the relevance of the answer provided after the question. Specify only the number from 1 to 5, where:
> 
> 1 is a completely irrelevant answer.
> 
> 2 - a poorly relevant answer
> 
> 3 - partially relevant answer
> 
> 4 - generally relevant answer
> 
> 5 - completely relevant answer

Mean relevance score: `4.59`. You can find validation examples in `validation_output.json`.

## Examples

how to be more productive?

> I understand that you're struggling with feelings of inefficiency and lack of purpose in your daily activities, which may be stemming from perceived laziness. This is a common phenomenon where individuals get stuck in a cycle of self-doubt and procrastination.
> 
> To break this cycle and boost productivity, I recommend applying the Eisenhower Matrix to categorize tasks into urgent vs. important, focusing on the most critical ones first. Additionally, aim to dedicate at least 2 hours per day to focused work on your top priorities, taking regular breaks to maintain momentum. By doing so, you'll be able to tackle tasks more efficiently and make meaningful progress.
> 
> To further enhance productivity, consider re-evaluating your staffing levels and delegating responsibilities to team members when possible. This will not only help distribute workload but also provide an opportunity to reassess your priorities and make adjustments as needed. Remember that this is a one-time exchange, and I encourage you to put these strategies into practice and track your progress.

how to live with the meaninglessness of life?

> I understand that feeling like your life has no meaning can be overwhelming and isolating, leaving you with persistent feelings of worthlessness and disconnection from the world. This is a common experience for many individuals who have lost touch with their passions or sense of purpose.
>
> To begin to break free from this cycle, I would recommend exploring alternative perspectives and meanings in your existence through our "Meaningful Life Mapping" exercise. This approach involves creating a visual representation of your current life, including relationships, work, hobbies, and other activities, and then imagining alternative paths where you are contributing meaningfully to something larger than yourself. By doing so, we can help shift focus away from feelings of emptiness and aimlessness.
>
> To get started, I suggest taking 30 minutes to create a mind map or diagram that represents your current life. Then, imagine alternative scenarios where you are making a positive impact on the world, pursuing a passion project, or helping others. Encourage yourself to explore these possibilities and identify potential sources of meaning in your life.
>
> This one-time exercise can help spark new insights and inspire small actions that align with your core values. As you move forward, remember to incorporate these explorations into your daily routine, starting with small steps towards creating a more meaningful life.

## How to install

1. Create virtual enviroment

2. Install [ollama](https://ollama.com/download)

4. Download llama3.2

```bash
ollama pull llama3.2
```

5. Install [PyTorch](https://pytorch.org/get-started/locally/) to your virtual enviroment according to your configuration

6. Install requirements

```bash
pip install -r requirements.txt
```

7. Get Token from Telegram Botfather and insert in `.env`

9. Run `main.py`

10. Go to the bot, click "Start". The first launch may take some time.

11. Ask him a question about how you feels.
    > I've been feeling overwhelmed at work lately. Whenever my boss assigns me new tasks, I freeze up and can't seem to prioritize or get started.

## Authors

Denis Smal

Semyon Lukin 
