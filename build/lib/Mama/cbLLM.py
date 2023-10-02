import logging
from langchain.llms import OpenAIChat, OpenAI
from config import Configuration 
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain import PromptTemplate

class cbLLM :
    def __init__(self):
        self.llm = None
        try:
            self._load()
        except Exception as e:
            logging.info("Error Loading LLM from configuration")
    
    def get_llm(self) :
        return self.llm
    
    def load_QueryChain(self, prompt) -> StuffDocumentsChain :
        if not self.llm:
            return None # type: ignore
        
        # This controls how each document will be formatted. Specifically,
        # it will be passed to `format_document` - see that function for more
        # details.
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )

        # The prompt here should take as an input variable the
        # `document_variable_name`
        prompt = PromptTemplate.from_template(
            "Summarize this content: {context}"
        )
        document_variable_name = "context"
        
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name
        )
        return chain
    
    def _load(self) :
        llms = {
            "OpenAi": OpenAI
        }
        config = Configuration()
        if not config:
            logging.info("Error Loading Configuration")
            return None
        
        model = config.get("model")
        if not model:
            logging.info("No LLM model found")
            return None

        llm_class = llms.get(model, None)
        if not llm_class:
            logging.info(f"model {model} not yet implemented!")
            return None
        
        params = config.get_llm_params(model)
        self.llm = llm_class(**params) 

        self.prompt_template = config.get_prompt_template()
        self.input_variables = config.get_input_variables()

    def get_prompt_template(self):
        return self.prompt_template
    
    def get_input_variables(self):
        return self.input_variables
            
    '''model_name: str = Field("text-davinci-003", alias="model")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    max_tokens: int = 256
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""
    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""
    frequency_penalty: float = 0
    """Penalizes repeated tokens according to frequency."""
    presence_penalty: float = 0
    """Penalizes repeated tokens."""
    n: int = 1
    """How many completions to generate for each prompt."""
    best_of: int = 1
    """Generates best_of completions server-side and returns the "best"."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_organization: Optional[str] = None
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    batch_size: int = 20
    """Batch size to use when passing multiple documents to generate."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to OpenAI completion API. Default is 600 seconds."""
    logit_bias: Optional[Dict[str, float]] = Field(default_factory=dict)
    """Adjust the probability of specific tokens being generated."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    allowed_special: Union[Literal["all"], AbstractSet[str]] = set()
    """Set of special tokens that are allowed。"""
    disallowed_special: Union[Literal["all"], Collection[str]] = "all"
    """Set of special tokens that are not allowed。"""
    tiktoken_model_name: Optional[str] = None'''

