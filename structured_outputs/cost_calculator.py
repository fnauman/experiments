#!/usr/bin/env python3
"""
OpenAI Vision API Cost Calculator for Garment Analysis
Comprehensive analysis of costs for processing images at scale
"""

from dataclasses import dataclass
from typing import Dict, List
import json

@dataclass
class CostBreakdown:
    scenario_name: str
    total_cost: float
    cost_per_image: float
    input_cost: float
    output_cost: float
    savings_vs_baseline: float
    savings_percentage: float
    monthly_cost_1k_daily: float

class VisionAPICostCalculator:
    def __init__(self):
        # Current OpenAI pricing (January 2025)
        self.pricing = {
            "gpt-4o-2024-08-06": {
                "input_per_1k": 0.0025,     # $2.50 per 1M tokens
                "output_per_1k": 0.010,     # $10.00 per 1M tokens  
                "cached_input_per_1k": 0.00125,  # 50% discount for cached tokens
                "batch_discount": 0.5,      # 50% discount for batch API
            }
        }
        
        # Token costs for different image detail levels
        self.image_tokens = {
            "low": 85,      # Fixed 85 tokens regardless of image size
            "auto": 1700,   # ~1700 tokens for 512x512 image
            "high": 2550,   # ~1.5x multiplier for high detail
        }
        
        # Typical prompt sizes for garment analysis
        self.prompt_sizes = {
            "system_prompt": 120,      # Detailed analysis instructions
            "user_prompt": 30,         # "Analyze this garment"
            "completion": 45,          # Structured JSON response
        }

    def calculate_single_call_cost(self, 
                                 image_detail: str = "auto",
                                 use_batch: bool = False,
                                 cache_system_prompt: bool = False) -> Dict:
        """Calculate cost for a single API call."""
        
        # Token calculations
        system_tokens = self.prompt_sizes["system_prompt"]
        user_tokens = self.prompt_sizes["user_prompt"] 
        image_tokens = self.image_tokens[image_detail]
        completion_tokens = self.prompt_sizes["completion"]
        
        # Input cost calculation
        if cache_system_prompt:
            cached_tokens = system_tokens
            fresh_tokens = user_tokens
        else:
            cached_tokens = 0
            fresh_tokens = system_tokens + user_tokens
            
        fresh_input_cost = (fresh_tokens / 1000) * self.pricing["gpt-4o-2024-08-06"]["input_per_1k"]
        cached_input_cost = (cached_tokens / 1000) * self.pricing["gpt-4o-2024-08-06"]["cached_input_per_1k"]
        image_cost = (image_tokens / 1000) * self.pricing["gpt-4o-2024-08-06"]["input_per_1k"]
        
        input_cost = fresh_input_cost + cached_input_cost + image_cost
        
        # Output cost
        output_cost = (completion_tokens / 1000) * self.pricing["gpt-4o-2024-08-06"]["output_per_1k"]
        
        # Total before batch discount
        subtotal = input_cost + output_cost
        
        # Apply batch discount
        if use_batch:
            batch_savings = subtotal * self.pricing["gpt-4o-2024-08-06"]["batch_discount"]
            total_cost = subtotal - batch_savings
        else:
            batch_savings = 0
            total_cost = subtotal
            
        return {
            "total_cost": total_cost,
            "input_cost": input_cost - (batch_savings * input_cost / subtotal) if subtotal > 0 else 0,
            "output_cost": output_cost - (batch_savings * output_cost / subtotal) if subtotal > 0 else 0,
            "image_tokens": image_tokens,
            "text_tokens": fresh_tokens + cached_tokens,
            "batch_savings": batch_savings,
            "cached_savings": (cached_tokens / 1000) * (
                self.pricing["gpt-4o-2024-08-06"]["input_per_1k"] - 
                self.pricing["gpt-4o-2024-08-06"]["cached_input_per_1k"]
            ) if cache_system_prompt else 0
        }

    def analyze_scenarios(self, num_images: int = 1000) -> List[CostBreakdown]:
        """Analyze different optimization scenarios."""
        
        scenarios = [
            ("Baseline (Individual, Auto Detail)", "auto", False, False),
            ("Individual + Prompt Caching", "auto", False, True),
            ("Batch API (No Caching)", "auto", True, False),
            ("Batch API + Prompt Caching", "auto", True, True),
            ("Low Detail + Batch + Caching", "low", True, True),
            ("High Detail + Batch + Caching", "high", True, True),
        ]
        
        results = []
        baseline_cost = None
        
        for name, detail, batch, cache in scenarios:
            single_cost = self.calculate_single_call_cost(detail, batch, cache)
            total_cost = single_cost["total_cost"] * num_images
            
            if baseline_cost is None:
                baseline_cost = total_cost
                savings = 0
                savings_pct = 0
            else:
                savings = baseline_cost - total_cost
                savings_pct = (savings / baseline_cost) * 100
            
            results.append(CostBreakdown(
                scenario_name=name,
                total_cost=total_cost,
                cost_per_image=single_cost["total_cost"],
                input_cost=single_cost["input_cost"] * num_images,
                output_cost=single_cost["output_cost"] * num_images,
                savings_vs_baseline=savings,
                savings_percentage=savings_pct,
                monthly_cost_1k_daily=total_cost * 30
            ))
        
        return results

    def calculate_scale_analysis(self) -> Dict:
        """Analyze costs at different scales."""
        scales = [10, 100, 1000, 10000, 100000]
        
        # Best case scenario: Batch + Caching + Auto detail
        single_cost = self.calculate_single_call_cost("auto", True, True)["total_cost"]
        
        analysis = {}
        for scale in scales:
            total_cost = single_cost * scale
            analysis[f"{scale}_images"] = {
                "total_cost": total_cost,
                "cost_per_image": single_cost,
                "daily_if_processed_monthly": total_cost / 30,
                "monthly_if_daily": total_cost * 30,
                "yearly_if_daily": total_cost * 365
            }
        
        return analysis

    def print_comprehensive_analysis(self):
        """Print a comprehensive cost analysis."""
        
        print("🔍 OPENAI VISION API - COMPREHENSIVE COST ANALYSIS")
        print("=" * 80)
        print("For Garment Analysis with Structured Outputs (GPT-4o-2024-08-06)")
        print()
        
        # Single image analysis
        print("💰 SINGLE IMAGE COST BREAKDOWN:")
        print("-" * 40)
        
        for detail in ["low", "auto", "high"]:
            cost = self.calculate_single_call_cost(detail, False, False)
            print(f"   {detail.capitalize():5} detail: ${cost['total_cost']:.4f} "
                  f"({cost['image_tokens']:4d} image tokens)")
        
        print()
        
        # Scenario comparison for 1000 images
        print("📊 COST SCENARIOS FOR 1,000 IMAGES:")
        print("-" * 60)
        
        scenarios = self.analyze_scenarios(1000)
        
        print(f"{'Scenario':<35} {'Total Cost':<12} {'Savings':<15} {'Per Image'}")
        print("-" * 75)
        
        for scenario in scenarios:
            savings_str = f"${scenario.savings_vs_baseline:.2f} ({scenario.savings_percentage:.1f}%)"
            if scenario.savings_vs_baseline == 0:
                savings_str = "Baseline"
            
            print(f"{scenario.scenario_name:<35} "
                  f"${scenario.total_cost:<11.2f} "
                  f"{savings_str:<15} "
                  f"${scenario.cost_per_image:.4f}")
        
        print()
        
        # Monthly projections
        print("📅 MONTHLY COST PROJECTIONS (1,000 images/day):")
        print("-" * 50)
        
        optimal_scenario = scenarios[-3]  # Batch + Caching with auto detail
        monthly_cost = optimal_scenario.monthly_cost_1k_daily
        
        print(f"   Daily processing cost:  ${optimal_scenario.total_cost:.2f}")
        print(f"   Monthly cost (30 days): ${monthly_cost:.2f}")
        print(f"   Yearly cost (365 days): ${monthly_cost * 12.17:.2f}")
        print()
        
        # Scale analysis
        print("📈 COST AT DIFFERENT SCALES (Optimized: Batch + Caching):")
        print("-" * 55)
        
        scale_analysis = self.calculate_scale_analysis()
        
        print(f"{'Scale':<15} {'Total Cost':<12} {'Monthly if Daily':<18}")
        print("-" * 45)
        
        for scale_key, data in scale_analysis.items():
            scale = scale_key.replace("_images", "")
            if int(scale) <= 100000:  # Only show reasonable scales
                print(f"{scale:<15} ${data['total_cost']:<11.2f} ${data['monthly_if_daily']:<17.2f}")
        
        print()
        
        # Recommendations
        print("🎯 RECOMMENDATIONS:")
        print("-" * 20)
        
        best_scenario = scenarios[3]  # Batch + Caching
        
        print(f"✅ FOR 1,000 IMAGES: Use Batch API + Prompt Caching")
        print(f"   💸 Save ${best_scenario.savings_vs_baseline:.2f} "
              f"({best_scenario.savings_percentage:.0f}%) vs individual calls")
        print(f"   📊 Monthly cost: ${best_scenario.monthly_cost_1k_daily:.2f}")
        print()
        print(f"✅ IMAGE DETAIL RECOMMENDATION: Use 'auto' detail")
        print(f"   🔄 Good balance of cost and quality")
        print(f"   💡 Use 'low' detail if basic classification is sufficient")
        print()
        print(f"✅ IMPLEMENTATION PRIORITY:")
        print(f"   1. Batch API implementation (saves ~50%)")
        print(f"   2. Prompt caching setup (additional savings)")
        print(f"   3. Image preprocessing (resize to 512x512)")
        print(f"   4. Monitor actual vs estimated token usage")
        
        print()
        print("💡 BREAK-EVEN ANALYSIS:")
        print("-" * 25)
        
        # Calculate when optimization becomes worthwhile
        individual_cost = scenarios[0].cost_per_image
        batch_cost = scenarios[3].cost_per_image
        savings_per_image = individual_cost - batch_cost
        
        # Assume 2-3 days development time at $100/hour
        development_cost = 2.5 * 8 * 100  # $2,000
        break_even_images = development_cost / savings_per_image
        
        print(f"   Development effort: ~2-3 days ($2,000)")
        print(f"   Break-even point: ~{break_even_images:,.0f} images")
        print(f"   At 1,000 images/day: Break-even in {break_even_images/1000:.1f} days")
        print(f"   ROI after 1 month: ${(30 * best_scenario.savings_vs_baseline) - development_cost:,.0f}")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    calculator = VisionAPICostCalculator()
    calculator.print_comprehensive_analysis() 