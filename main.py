import openai
import os
from dotenv import load_dotenv

prompt = """A list of ideas for products or services that are terrible or funny and don't yet exist:
- Website that diagnoses you with the worst possible condition
- API that tells you if a number is even
- Instant noodles milkshake
- Waterproof toaster
- Sneakers that look like socks
- Laundry detergent that doubles as a facial cleanser
- """

load_dotenv()

def filterContent(content_to_classify: str) -> bool:
    # Make request to OpenAI filter API
    # The filter is free to use.
    response = openai.Completion.create(
      engine="content-filter-alpha-c4",
      prompt = "<|endoftext|>"+content_to_classify+"\n--\nLabel:",
      temperature=0,
      max_tokens=1,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      logprobs=10
    )
    output_label = response["choices"][0]["text"]

    # This is the probability at which we evaluate that a "2" is likely real
    # vs. should be discarded as a false positive
    toxic_threshold = -0.355

    if output_label == "2":
        # If the model returns "2", return its confidence in 2 or other output-labels
        logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]

        # If the model is not sufficiently confident in "2",
        # choose the most probable of "0" or "1"
        # Guaranteed to have a confidence for 2 since this was the selected token.
        if logprobs["2"] < toxic_threshold:
            logprob_0 = logprobs.get("0", None)
            logprob_1 = logprobs.get("1", None)

            # If both "0" and "1" have probabilities, set the output label
            # to whichever is most probable
            if logprob_0 is not None and logprob_1 is not None:
                if logprob_0 >= logprob_1:
                    output_label = "0"
                else:
                    output_label = "1"
            # If only one of them is found, set output label to that one
            elif logprob_0 is not None:
                output_label = "0"
            elif logprob_1 is not None:
                output_label = "1"

            # If neither "0" or "1" are available, stick with "2"
            # by leaving output_label unchanged.

    # if the most probable token is none of "0", "1", or "2"
    # this should be set as unsafe
    if output_label not in ["0", "1", "2"]:
        output_label = "2"

    return output_label != "2"

def generateIdea(recursiveDepth=0):
    # Load in API key from either .env file or environment variables
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Generate Idea using Davinci

    completion = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=1,
        max_tokens=24,
        top_p=0.59,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop=["\n"]
    )
    generated_idea = completion["choices"][0]["text"]

    # Filter Unsafe content, as per guidelines
    if not filterContent(generated_idea) and len(generated_idea) > 0:
        print("Model generated unsafe idea")
        return generateIdea(recursiveDepth+1)
    




    return generated_idea

if __name__ == "__main__":
    while True:
        print(generateIdea())
        input()