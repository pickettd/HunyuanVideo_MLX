from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ModelConfig:
    """Configuration for a single model in the pipeline"""
    name: str  # Name of the model (e.g., "model_540p", "model_720p")
    checkpoint_path: str  # Path to model checkpoint
    start_step: int  # Starting step for this model
    end_step: int  # Ending step for this model
    description: str  # Description of what this model does

@dataclass
class MMGPConfig:
    """Configuration for Multi-Model Generation Pipeline"""
    models: Dict[str, ModelConfig]  # Dictionary of model configurations
    schedule: List[Dict]  # Schedule defining when to use each model
    memory_optimization: Dict  # Memory optimization settings
    quality_tips: Dict  # Tips for quality settings
    performance: Dict  # Performance settings for different hardware

    @classmethod
    def from_json(cls, config_path: str) -> "MMGPConfig":
        """Load MMGP configuration from a JSON file"""
        import json
        with open(config_path) as f:
            data = json.load(f)
            
        models = {}
        for model_name, checkpoint_path in data.get("models", {}).items():
            models[model_name] = ModelConfig(
                name=model_name,
                checkpoint_path=checkpoint_path,
                start_step=0,  # Will be updated from schedule
                end_step=0,  # Will be updated from schedule
                description=""  # Will be updated from schedule
            )
            
        # Update model configs from schedule
        schedule = data.get("schedule", [])
        for step in schedule:
            model_name = step["model"]
            if model_name in models:
                models[model_name].start_step = step["start_step"]
                models[model_name].end_step = step["end_step"]
                models[model_name].description = step.get("description", "")
                
        return cls(
            models=models,
            schedule=schedule,
            memory_optimization=data.get("notes", {}).get("memory_optimization", {}),
            quality_tips=data.get("notes", {}).get("quality_tips", {}),
            performance=data.get("notes", {}).get("performance", {})
        )

    def get_model_for_step(self, step: int) -> Optional[ModelConfig]:
        """Get the appropriate model configuration for a given step"""
        for model_config in self.models.values():
            if model_config.start_step <= step < model_config.end_step:
                return model_config
        return None

    def validate(self) -> bool:
        """Validate the configuration"""
        # Check that all models exist
        for schedule_item in self.schedule:
            model_name = schedule_item["model"]
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} referenced in schedule but not defined in models")
                
        # Check that steps are continuous and non-overlapping
        sorted_schedule = sorted(self.schedule, key=lambda x: x["start_step"])
        for i in range(len(sorted_schedule) - 1):
            current = sorted_schedule[i]
            next_item = sorted_schedule[i + 1]
            if current["end_step"] != next_item["start_step"]:
                raise ValueError(
                    f"Gap or overlap in schedule between steps {current['end_step']} "
                    f"and {next_item['start_step']}"
                )
                
        return True
