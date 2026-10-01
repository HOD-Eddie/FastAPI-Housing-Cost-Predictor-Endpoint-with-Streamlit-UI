from transformers import pipeline

class DescriptionService:
    def __init__(self):
        print("--> Initializing Pre-trained LLM Pipeline (Loading distilgpt2)...")
        # 1. Download and load the pre-trained text generation engine
        # This will download the model exactly once on boot and cache it locally
        self.generator = pipeline("text-generation", model="distilgpt2")
        print("--> Pre-trained LLM Pipeline successfully loaded.")

    def generate_listing(self, input_data: dict, predicted_price: float) -> str:
        """
        Takes the input features and the price from Station 1, constructs
        a precise prompt engineering template, and asks the LLM to generate sales copy.
        """
        # 2. Build an intuitive template mapping out the variables
        prompt = (
            f"Real estate listing description for a beautiful home with {input_data['avg_rooms']:.1f} rooms. "
            f"The property is located in an area with an average income of ${input_data['median_income']*10000:.0f}. "
            f"It is officially valued at a fair market price of ${predicted_price:,.2f}. "
            f"Write an engaging, {input_data['marketing_tone']} advertisement script for this house:\n\n"
            "Welcome to your dream home!"
        )

        # 3. Feed the prompt string to the pre-trained model
        # max_new_tokens limits how long the generated text can be
        # temperature controls creativity (0.7 is a solid blend of logic and flare)
        response = self.generator(
            prompt, 
            max_new_tokens=60, 
            do_sample=True,         # Enables creative sampling
            temperature=0.8,        # Slightly higher for more flavor
            repetition_penalty=1.2, # Stops word loops!
            num_return_sequences=1,
            pad_token_id=50256
    )

        # 4. Extract the raw text result array
        full_generated_text = response[0]["generated_text"]

        # Clean up: Since the generator returns your prompt back along with the new text,
        # we can strip out the prompt part so the user only gets the clean marketing script.
        clean_description = full_generated_text.replace(prompt, "Welcome to your dream home! ")
        
        return clean_description.strip()

# Instantiate the service singleton instance for imports
description_service = DescriptionService()
