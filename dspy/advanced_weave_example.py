import os
import dspy
import weave
from dotenv import load_dotenv


load_dotenv() # OPENROUTER_API_KEY is stored in .env file


weave.init(project_name="dspy-intro")

gpt3_turbo = dspy.OpenAI(model='gpt-3.5-turbo-1106', max_tokens=300)
dspy.configure(lm=gpt3_turbo)

class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context = dspy.InputField(desc="facts here are assumed to be true")
    text = dspy.InputField()
    faithfulness = dspy.OutputField(desc="True/False indicating if text is faithful to context")


class WeaveModel(weave.Model):
    signature: type

    @weave.op()
    def predict(self, context: str, text: str) -> bool:
        return dspy.ChainOfThought(self.signature)(context=context, text=text)

context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."
text = "Lee scored 3 goals for Colchester United."

model = WeaveModel(signature=CheckCitationFaithfulness)
print(model.predict(context, text))
