"""
demo_tools.py

Wraps the existing CRUCIBLE tools for use with the LLM.
This defines the tool schema (what the LLM sees) and execution logic.
"""

import json
from tools import identify_material  # Your existing function

# =============================================================================
# TOOL SCHEMA - This tells the LLM what tools are available
# =============================================================================

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "identify_material",
        "description": "Identify a material based on its Raman spectroscopy peaks and formation energy. Returns the predicted material name with confidence score.",
        "parameters": {
            "type": "object",
            "properties": {
                "peak_1": {
                    "type": "number",
                    "description": "First Raman peak in wavenumbers (cm^-1). Typically between 100-2000 cm^-1."
                },
                "peak_2": {
                    "type": "number",
                    "description": "Second Raman peak in wavenumbers (cm^-1). Typically between 100-2000 cm^-1."
                },
                "formation_energy": {
                    "type": "number",
                    "description": "Formation energy in eV per atom. Typically between -15 and 0 eV/atom. Negative values indicate stable compounds."
                }
            },
            "required": ["peak_1", "peak_2", "formation_energy"]
        }
    }
}


# =============================================================================
# TOOL EXECUTION - This actually runs your tool
# =============================================================================

def execute_tool(tool_name, arguments):
    """
    Execute a tool function with given arguments.
    
    Args:
        tool_name: Name of the tool to execute (should be "identify_material")
        arguments: Dictionary with keys: peak_1, peak_2, formation_energy
    
    Returns:
        JSON string with the result
    """
    if tool_name == "identify_material":
        try:
            # Call your existing function
            result = identify_material(
                arguments["peak_1"],
                arguments["peak_2"],
                arguments["formation_energy"]
            )
            
            # Return as JSON for the LLM
            return json.dumps({
                "success": True,
                "result": result
            })
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    else:
        return json.dumps({
            "success": False,
            "error": f"Unknown tool: {tool_name}"
        })


# =============================================================================
# TEST CODE - Run this file directly to test the tool
# =============================================================================

if __name__ == "__main__":
    print("Testing demo_tools.py")
    print("=" * 60)
    
    # Test with known Ceria values
    print("\n1. Testing identify_material with Ceria values:")
    print("   Peak 1: 465 cm^-1")
    print("   Peak 2: 610 cm^-1")
    print("   Formation Energy: -11.2 eV/atom")
    
    result = execute_tool(
        "identify_material",
        {
            "peak_1": 465,
            "peak_2": 610,
            "formation_energy": -11.2
        }
    )
    
    print("\n   Result:")
    result_dict = json.loads(result)
    print(f"   {result_dict}")
    
    print("\n" + "=" * 60)
    print(" Tool test complete!")
    print("\nNext: Run demo_llm.py to chat with CRUCIBLE!")
