from abc import abstractmethod
from typing import Optional, Union,List,Dict
import itertools

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding

"""
Classes for zero/few-shot text classification, with HuggingFace models.

Two methods are implemented:
- 'Cloze' based on Shick et. al. (2020)

Example usage:

    input_texts = [
        "This was terrible service",
        "I want to terminate the policy."
    ]
    label_words = [["positive","neutral"], ["negative"]]

    cloze_template = ClozeTemplate(["email:", "this email has {} sentiment"])
    cloze_wrapper = ClozeWrapper(
        mlm_model_tokenizer,
        cloze_template,
        label_words
    )
    cloze_preds = cloze_wrapper(mlm_model, input_texts)

"""

class ClozeTemplate():
    """Prompt template for the cloze method.

    A prompt template is given on the form [prepended_str, appended_str],
    where either prepended_str or appended_str contains a single
    empty placeholder {} representing the masked position.
    The resulting prompt of an input x is:
    prompt(x) = prepended_str + x + appended_str

    Example:
        The prompt template ["", "This text is about {}."]
        produces the prompt "[input text] This text is about [MASK]."
    """
    def __init__(self, template: List[str]):
        assert len(template) == 2

        masked_position_token = "{}"
        n_masked_position_token = sum([
            text_snippet.count(masked_position_token)
            for text_snippet in template
        ])
        assert n_masked_position_token == 1

        self.template = template

    def __repr__(self) -> str:
        template = [s.format("_") for s in self.template]
        return ''.join(['[', template[0], '][', template[1], ']'])
    

class Wrapper:
    """Zero/few-shot classification wrapper for usage with HuggingFace.

    Uses a prompt template and a list of label words to produce
    input to the underlying language model, and to parse its outputs.

    The label words define a one-to-one mapping between the tokens of the
    language model and the classes of the classification problem.
    The prompt template is used to reformulate the classification problem
    as an instance of the NLP problem the language model has been trained on.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, template, label_words):
        self.tokenizer = tokenizer
        self.init_label_words(label_words)
        self.init_template(template)
        self.device = self.get_device()
        
    def get_device(self):
        if torch.cuda.is_available():
            return "cuda:0"  # Use only one GPU.
        return "cpu"

    @classmethod
    @abstractmethod
    def init_template(cls, template):
        """Initialize prompt template."""

    @classmethod
    @abstractmethod
    def init_label_words(cls, label_words):
        """Initialize label words."""

    @classmethod
    @abstractmethod
    def preprocess_inputs(cls, self, inputs, *args, **kwargs):
        """Preprocess a batch of text inputs.

        Useful for data loader pipelines.
        """

    @classmethod
    @abstractmethod
    def forward(cls, model, inputs, softmax_output):
        """Forward pass using inputs produced by preprocess_inputs."""

    @classmethod
    @abstractmethod
    def __call__(cls, model, inputs):
        """Compute output probabilities of a batch of input texts."""

    def to_device(self, inputs):
        """Put inputs on device. """
        return {
            name: tensor.to(self.device)
            for name, tensor in inputs.items()
        }


class ClozeWrapper(Wrapper):
    """Wrapper for the Cloze Method (Schick et al. 2021).

    Text classification is reframed as a MLM problem.
    An input text is mapped to a prompt by a prompt template,
    which adds text containing a single [MASK] token to it.

    Reference:
    https://aclanthology.org/2021.eacl-main.20.pdf
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 template: ClozeTemplate,
                 label_words: Optional[List[str]] = None):
        super().__init__(tokenizer, template, label_words)
        self.max_seq_len = self.compute_max_seq_len()

    def init_template(self, template: List[str]):
        """Initialize the prompt template.

        The prompt template is saved in its tokenized format.

        Args:
            template: The prompt template to use.
        """

        self.template = []
        for text_snippet in template:
            masked_text_snippet = text_snippet.format(
                self.tokenizer.mask_token)
            tokenized_text_snippet = self.tokenizer.encode(
                masked_text_snippet,
                add_special_tokens=False)
            self.template.append(tokenized_text_snippet)

    def compute_max_seq_len(self) -> int:
        """Compute the maximum sequence length of an input sequence.

        Large input sequences need to be truncated while keeping the
        prompt template untruncated.

        Returns:
            The maximum sequence length in number of tokens.
        """
        template_len = len(self.template[0]) + len(self.template[1])
        n_special_chars = self.tokenizer.num_special_tokens_to_add(False)

        max_seq_len = self.tokenizer.model_max_length \
            - (template_len + n_special_chars)

        assert max_seq_len > 0

        return max_seq_len

    def init_label_words(self, label_words: Optional[List[str]]):
        """Initialize label words.

        The entire vocabulary of the tokenizer is used as
        label words if no label words are specified.

        Args:
            label_words: List of label words to use.
        """

        if label_words is None:
            # Use full vocabulary.
            self.label_word_ids = [x for x in range(self.tokenizer.vocab_size)]
            self.n_classes = self.tokenizer.vocab_size
        else:
            self.n_classes = len(label_words)
            self.label_word_ids=[[] for _ in range(self.n_classes)]
            for i, label in enumerate(label_words):
                for char in label:
                    tokenized_label = self.tokenizer(char, add_special_tokens=False)['input_ids']
                    self.label_word_ids[i].append(tokenized_label[0])                

    def gen_prompts(self, inputs: List[List[int]], *args,
                    **kwargs) -> BatchEncoding:
        """Generate prompts.

        Args:
            inputs: Batch of inputs represented as lists of token ids.

        Returns:
            Prompts generated from inputs (as lists of token ids).
        """

        prompts = []
        for inp in inputs:
            truncated_input = inp[:self.max_seq_len]

            prompt = itertools.chain(self.template[0], truncated_input,self.template[1])
            
            prompts.append(list(prompt))

        # tokenized_prompts = self.tokenizer.batch_encode_plus(
        #     prompts, is_split_into_words=True, *args, **kwargs)
        # return tokenized_prompts
        return prompts

    def preprocess_inputs(self, inputs: Union[List[str], List[int]], *args,
                          **kwargs) -> BatchEncoding:
        """Preprocesses inputs by mapping them to their prompts.

        Args:
            inputs: List of input strings or lists of token ids.

        Returns:
            Prompts represented as lists of token ids.
        """

        is_not_pretokenized = isinstance(inputs[0], str)

        if is_not_pretokenized:
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                inputs,
                add_special_tokens=False,
                return_attention_mask=False,
                truncation=True)['input_ids']
        else:
            tokenized_inputs = inputs

        prompts = self.gen_prompts(tokenized_inputs, *args, **kwargs)

        return prompts

    def parse_outputs(self,
                      outputs: torch.Tensor,
                      masked_idx: torch.Tensor,
                      softmax_output=True) -> torch.Tensor:
        """Parse outputs of a MLM model.

        The MLM model outputs a probability distribution over the entire
        vocabulary for each element in an input sequence.
        This method convert the outputs associated with an
        input text to an output of the text classification problem.

        Args:
            outputs: A batch of outputs produced by the MLM model.
            masked_idx: Indices of masked positions for each output.
            softmax_output: True if softmax should be computed over the outputs.

        Returns:
            Tensor with output predictions over the classes of interest.
        """

        logit=torch.zeros(outputs.shape[0],len(self.label_word_ids),dtype=torch.float)
        for i,label_ids in enumerate(self.label_word_ids):
            x=0
            for ids in label_ids:
                x+=outputs[:, :, ids][:,masked_idx]
            logit[:,i]=x
        if softmax_output:
            logit=logit.softmax(dim=-1)

        return logits

    def get_masked_idx(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get position of masked indices in inputs."""
        return torch.where(
            inputs['input_ids'] == self.tokenizer.mask_token_id)[1]

    def forward(self,
                model: PreTrainedModel,
                inputs: BatchEncoding,
                softmax_output=True) -> torch.Tensor:

        prompts = self.to_device(inputs)
        masked_idx = self.get_masked_idx(prompts)
        outputs = model(**prompts).logits
        parsed_outputs = self.parse_outputs(outputs, masked_idx,
                                            softmax_output)

        return parsed_outputs

    def __call__(self, model, inputs, softmax_output=True):
        with torch.no_grad():
            inputs = self.preprocess_inputs(inputs,
                                            return_tensors="pt",
                                            padding=True)
            outputs = self.forward(model, inputs, softmax_output)
        return outputs
    
    