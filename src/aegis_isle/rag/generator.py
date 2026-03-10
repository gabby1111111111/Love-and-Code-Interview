"""
Text generation components for RAG pipeline.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..core.config import settings
from ..core.logging import logger
from .retriever import EnhancedQueryResult


class GenerationConfig(BaseModel):
    """Configuration for text generation."""

    model: str = "gpt-4-1106-preview"
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None


class GenerationResult(BaseModel):
    """Result of text generation."""

    generated_text: str
    model: str
    usage: Dict[str, Any] = {}
    generation_time: float = 0.0
    metadata: Dict[str, Any] = {}


class BaseGenerator(ABC):
    """Base class for text generators."""

    def __init__(self, config: GenerationConfig):
        self.config = config

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        retrieval_context: Optional[EnhancedQueryResult] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate text based on prompt and optional context."""
        pass

    @abstractmethod
    async def generate_with_context(
        self,
        query: str,
        context_chunks: List[str],
        **kwargs
    ) -> GenerationResult:
        """Generate text with provided context chunks."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        retrieval_context: Optional[EnhancedQueryResult] = None,
        **kwargs
    ):
        """Generate text stream based on prompt and optional context."""
        pass


class LLMGenerator(BaseGenerator):
    """LLM-based text generator supporting multiple providers."""

    def __init__(
        self,
        config: GenerationConfig,
        provider: str = "openai"
    ):
        super().__init__(config)
        self.provider = provider
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the LLM client based on provider."""
        try:
            if self.provider == "openai":
                from openai import AsyncOpenAI

                # 构建OpenAI客户端配置
                client_kwargs = {"api_key": settings.openai_api_key}

                # 如果配置了自定义base_url（如SiliconFlow等），使用它
                if settings.openai_base_url:
                    client_kwargs["base_url"] = settings.openai_base_url
                    logger.info(f"Using custom OpenAI base URL: {settings.openai_base_url}")

                self._client = AsyncOpenAI(**client_kwargs)
                self._generate_method = self._generate_openai

            elif self.provider == "anthropic":
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
                self._generate_method = self._generate_anthropic

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            logger.info(f"Initialized {self.provider} generator")

        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} generator: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        retrieval_context: Optional[EnhancedQueryResult] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate text with optional retrieval context."""
        start_time = time.time()

        try:
            # Build enhanced prompt if context is provided
            if retrieval_context and retrieval_context.results:
                enhanced_prompt = self._build_context_prompt(prompt, retrieval_context)
            else:
                enhanced_prompt = prompt

            # Generate response
            result = await self._generate_method(enhanced_prompt, **kwargs)

            # Add timing and metadata
            result.generation_time = time.time() - start_time
            result.metadata.update({
                "provider": self.provider,
                "has_context": bool(retrieval_context and retrieval_context.results),
                "context_chunks": len(retrieval_context.results) if retrieval_context else 0
            })

            return result

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerationResult(
                generated_text=f"Generation failed: {str(e)}",
                model=self.config.model,
                generation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def generate_stream(
        self,
        prompt: str,
        retrieval_context: Optional[EnhancedQueryResult] = None,
        **kwargs
    ):
        """Stream text generation with optional retrieval context."""
        # Build enhanced prompt if context is provided
        if retrieval_context and retrieval_context.results:
            enhanced_prompt = self._build_context_prompt(prompt, retrieval_context)
        else:
            enhanced_prompt = prompt

        # Merge config with kwargs
        generation_config = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
            "stream": True,
        }

        if self.config.stop_sequences:
            generation_config["stop"] = self.config.stop_sequences

        # We do not catch exceptions here to let the caller handle them
        logger.info(f"Calling LLM with provider={self.provider}, model={generation_config.get('model')}")
        logger.debug(f"Enhanced prompt: {enhanced_prompt[:200]}...")
        logger.debug(f"Generation config: {generation_config}")
        
        if self.provider == "openai":
            stream = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": enhanced_prompt}],
                **generation_config
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        elif self.provider == "anthropic":
            # Anthropic streaming implementation
            kwargs_copy = generation_config.copy()
            kwargs_copy.pop("stream", None) # Remove stream arg as we use stream() method
            # Adjust parameter names
            if "model" in kwargs_copy:
                 kwargs_copy["model"] = kwargs_copy["model"].replace("gpt", "claude")
            
            async with self._client.messages.stream(
                messages=[{"role": "user", "content": enhanced_prompt}],
                **kwargs_copy
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        else:
             raise ValueError(f"Streaming not supported for provider: {self.provider}")

    async def generate_with_context(
        self,
        query: str,
        context_chunks: List[str],
        **kwargs
    ) -> GenerationResult:
        """Generate text with provided context chunks."""
        # Create a mock EnhancedQueryResult for consistency
        from .retriever import RetrievalResult, EnhancedQueryResult
        from .document_processor import DocumentChunk

        mock_results = []
        for i, chunk_text in enumerate(context_chunks):
            chunk = DocumentChunk(
                document_id="provided_context",
                content=chunk_text,
                chunk_index=i
            )
            result = RetrievalResult(chunk=chunk, score=1.0)
            mock_results.append(result)

        context = EnhancedQueryResult(
            query=query,
            results=mock_results,
            total_time=0.0
        )

        return await self.generate(query, context, **kwargs)

    def _build_context_prompt(self, query: str, context: EnhancedQueryResult) -> str:
        """Build a prompt that includes retrieved context."""
        context_text = "\n\n".join([
            f"Context {i+1}:\n{result.chunk.content}"
            for i, result in enumerate(context.results)
        ])

        prompt_template = """You are an AI assistant helping to answer questions based on the provided context.

Context Information:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing and provide the best answer you can based on the available context.

Answer:"""

        return prompt_template.format(context=context_text, query=query)

    async def _generate_openai(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text using OpenAI API."""
        try:
            # Merge config with kwargs
            generation_config = {
                "model": kwargs.get("model", self.config.model),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
                "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
            }

            if self.config.stop_sequences:
                generation_config["stop"] = self.config.stop_sequences

            # Create chat completion
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **generation_config
            )

            return GenerationResult(
                generated_text=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    async def _generate_anthropic(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text using Anthropic API."""
        try:
            # Anthropic uses different parameter names
            generation_config = {
                "model": kwargs.get("model", self.config.model.replace("gpt", "claude")),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
            }

            if self.config.stop_sequences:
                generation_config["stop_sequences"] = self.config.stop_sequences

            # Create message
            response = await self._client.messages.create(
                messages=[{"role": "user", "content": prompt}],
                **generation_config
            )

            return GenerationResult(
                generated_text=response.content[0].text,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }
            )

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise


class HuggingFaceGenerator(BaseGenerator):
    """Local Hugging Face model generator."""

    def __init__(
        self,
        config: GenerationConfig,
        model_name: str = "microsoft/DialoGPT-medium",
        device: str = "auto"
    ):
        super().__init__(config)
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize Hugging Face model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )

            # Add pad token if missing
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            logger.info(f"Initialized Hugging Face model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face model: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        retrieval_context: Optional[EnhancedQueryResult] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate text using local Hugging Face model."""
        start_time = time.time()

        try:
            # Build enhanced prompt if context is provided
            if retrieval_context and retrieval_context.results:
                enhanced_prompt = self._build_context_prompt(prompt, retrieval_context)
            else:
                enhanced_prompt = prompt

            # Tokenize input
            inputs = self._tokenizer.encode(enhanced_prompt, return_tensors="pt").to(self.device)

            # Generate
            max_length = min(
                len(inputs[0]) + kwargs.get("max_tokens", self.config.max_tokens),
                self._model.config.max_length or 2048
            )

            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=kwargs.get("temperature", self.config.temperature),
                    top_p=kwargs.get("top_p", self.config.top_p),
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            # Decode output
            generated_text = self._tokenizer.decode(
                outputs[0][len(inputs[0]):],
                skip_special_tokens=True
            )

            return GenerationResult(
                generated_text=generated_text.strip(),
                model=self.model_name,
                generation_time=time.time() - start_time,
                usage={
                    "input_tokens": len(inputs[0]),
                    "output_tokens": len(outputs[0]) - len(inputs[0]),
                    "total_tokens": len(outputs[0])
                },
                metadata={"provider": "huggingface", "device": self.device}
            )

        except Exception as e:
            logger.error(f"Hugging Face generation failed: {e}")
            return GenerationResult(
                generated_text=f"Generation failed: {str(e)}",
                model=self.model_name,
                generation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def generate_with_context(
        self,
        query: str,
        context_chunks: List[str],
        **kwargs
    ) -> GenerationResult:
        """Generate text with provided context chunks."""
        # Similar to LLMGenerator implementation
        from .retriever import RetrievalResult, EnhancedQueryResult
        from .document_processor import DocumentChunk

        mock_results = []
        for i, chunk_text in enumerate(context_chunks):
            chunk = DocumentChunk(
                document_id="provided_context",
                content=chunk_text,
                chunk_index=i
            )
            result = RetrievalResult(chunk=chunk, score=1.0)
            mock_results.append(result)

        context = EnhancedQueryResult(
            query=query,
            results=mock_results,
            total_time=0.0
        )

        return await self.generate(query, context, **kwargs)

    def _build_context_prompt(self, query: str, context: EnhancedQueryResult) -> str:
        """Build a prompt that includes retrieved context."""
        context_text = "\n\n".join([
            f"Context {i+1}: {result.chunk.content}"
            for i, result in enumerate(context.results)
        ])

        # Simpler prompt for smaller models
        prompt_template = """Context:\n{context}\n\nQuestion: {query}\nAnswer:"""
        return prompt_template.format(context=context_text, query=query)


def get_generator(
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs
) -> BaseGenerator:
    """Factory function to get a generator based on provider."""
    model = model or settings.default_llm_model
    kwargs.pop('max_tokens', None)
    kwargs.pop('temperature', None)
    config = GenerationConfig(
        model=model,
        max_tokens=kwargs.get("max_tokens", settings.max_tokens),
        temperature=kwargs.get("temperature", settings.temperature),
        **kwargs
    )

    if provider == "openai":
        return LLMGenerator(config, provider="openai")
    elif provider == "anthropic":
        return LLMGenerator(config, provider="anthropic")
    elif provider == "huggingface":
        return HuggingFaceGenerator(config, model_name=model)
    else:
        logger.warning(f"Unknown provider '{provider}', defaulting to OpenAI")
        return LLMGenerator(config, provider="openai")