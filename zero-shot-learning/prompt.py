from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoModelForMaskedLM , AutoTokenizer
from typing import Optional, Union,List,Dict
import torch

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
    

class Prompting(object):
    """ doc string 
    This class helps us to implement
    Prompt-based Learning Model
    """
    def __init__(self, tokenizer:PreTrainedTokenizer, template:ClozeTemplate,  label_words: Optional[List[List[str]]]):
        """ constructor 
        parameter:
        ----------
           model: AutoModelForMaskedLM
                path to a Pre-trained language model form HuggingFace Hub
           tokenizer: AutoTokenizer
                path to tokenizer if different tokenizer is used, 
                otherwise leave it empty
        """
        self.tokenizer=tokenizer
        self.init_template(template)
        self.init_label_words(label_words)
        self.device=self.get_device()
        
        self.max_seq_len = self.compute_max_seq_len()
        
    def get_device(self):
        if torch.cuda.is_available():
            return "cuda:0"  # Use only one GPU.
        return "cpu"
    
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

        max_seq_len = self.tokenizer.model_max_length - (template_len + n_special_chars)

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
            self.label_word_ids=[[] for _ in range(n_classes)]
            for i, label in enumerate(label_words):
                for char in label:
                    tokenized_label = self.tokenizer(char, add_special_tokens=False)['input_ids']
                    self.label_word_ids[i].append(tokenized_label[0])
                    
    def gen_prompts(self, inputs: Union(List[List[int]], List[List[str]])) -> BatchEncoding:
        """Generate prompts.

        Args:
            inputs: Batch of inputs represented as lists of token ids.

        Returns:
            Prompts generated from inputs (as lists of token ids).
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

        prompts = []
        for inp in inputs:
            truncated_input = inp[:self.max_seq_len]

            prompt = itertools.chain(self.template[0], truncated_input,self.template[1])
            
            prompts.append(list(prompt))

        tokenized_prompts = self.tokenizer([tokenizer.decode(x) for x in prompts])
        return tokenized_prompts

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
        if len(self.label_word_ids)==self.tokenizer.vocab_size:
            values,indices=torch.sort(outputs[:, masked_idx],  descending=True,dim=-1)
            result=list(zip(self.tokenizer.convert_ids_to_tokens(indices), values))
            self.scores_dict={a:b for a,b in result}
            return result
        else:
            
            logit=torch.zeros(outputs.shape[0],len(self.label_word_ids),dtype=torch.float)
            for i,label_ids in enumerate(self.label_word_ids):
                x=0
                for ids in label_ids:
                    x+=outputs[:, :, ids][:,masked_idx]
                logit[:,i]=x
            if softmax_output:
                logit=logit.softmax(dim=-1)

            return logits
    
    def forward(self,
               inputs:Optional[BatchEncoding],
               model: PreTrainedModel,
               softmax_output=True):
        prompts=self.gen_prompts(inputs)
        mask_id=[x.index(self.tokenizer.mask_token_id) for x in prompts]
        
        model.eval()
        with torch.no_grad():
            outputs = model(**prompts).logits
        parsed_outputs = self.parse_outputs(outputs, masked_idx, softmax_output)
        
        return  parsed_outputs
                    
    def prompt_pred(self,text):
        """
          Predict MASK token by listing the probability of candidate tokens 
          where the first token is the most likely
          Parameters:
          ----------
          text: str 
              The text including [MASK] token.
              It only supports single MASK token. If more [MASK]ed tokens 
              are given, it takes the first one.
          Returns:
          --------
          list of (token, prob)
             The return is a list of all token in LM Vocab along with 
             their prob score, sort by score in descending order 
        """
        
        indexed_tokens=self.tokenizer(text, return_tensors="pt").input_ids
        tokenized_text= self.tokenizer.convert_ids_to_tokens(indexed_tokens[0])
        # take the first masked token
        mask_pos=tokenized_text.index(self.tokenizer.mask_token)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(indexed_tokens)
            predictions = outputs[0]
        values, indices=torch.sort(predictions[0, mask_pos],  descending=True)
        #values=torch.nn.functional.softmax(values, dim=0)
        result=list(zip(self.tokenizer.convert_ids_to_tokens(indices), values))
        self.scores_dict={a:b for a,b in result}
        return result

    def compute_tokens_prob(self, text, token_list1, token_list2):
        """
        Compute the activations for given two token list, 
        Parameters:
        ---------
        token_list1: List(str)
         it is a list for positive polarity tokens such as good, great. 
        token_list2: List(str)
         it is a list for negative polarity tokens such as bad, terrible.      
        Returns:
        --------
        Tuple (
           the probability for first token list,
           the probability of the second token list,
           the ratio score1/ (score1+score2)
           The softmax returns
           )
        """
        _=self.prompt_pred(text)
        score1=[self.scores_dict[token1] if token1 in self.scores_dict.keys() else 0\
                for token1 in token_list1]
        score1= sum(score1)
        score2=[self.scores_dict[token2] if token2 in self.scores_dict.keys() else 0\
                for token2 in token_list2]
        score2= sum(score2)
        
        softmax_rt=torch.nn.functional.softmax(torch.Tensor([score1,score2]), dim=0)
        return softmax_rt        
    
    def fine_tune(self, sentences, labels, prompt="it was [MASK] sentiment.",goodToken="positive",badToken="negative"):
        """  
          Fine tune the model
        """
        good=tokenizer.convert_tokens_to_ids(goodToken)
        bad=tokenizer.convert_tokens_to_ids(badToken)

        from transformers import AdamW
        optimizer = AdamW(self.model.parameters(),lr=1e-3)

        for sen, label in zip(sentences, labels):
            tokenized_text = self.tokenizer.tokenize(sen+prompt)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            # take the first masked token
            mask_pos=tokenized_text.index(self.tokenizer.mask_token)
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]
            pred=predictions[0, mask_pos][[good,bad]]
            prob=torch.nn.functional.softmax(pred, dim=0)
            lossFunc = torch.nn.CrossEntropyLoss()
            loss=lossFunc(prob.unsqueeze(0), torch.tensor([label]))
            loss.backward()
            optimizer.step()
        print("done!")
        
        