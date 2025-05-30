import json
import os

import gradio as gr
from gradio.components import Component
from typing import Any, Dict, Optional
from src.webui.webui_manager import WebuiManager
from src.utils import config
from src.utils.mcp_client import setup_mcp_client_and_tools
import logging
from functools import partial

logger = logging.getLogger(__name__)


async def update_model_dropdown(llm_provider, prog=gr.Progress(track_tqdm=True)):
    """
    Update the model name dropdown with dynamically fetched models for the selected provider.
    
    Args:
        llm_provider (str): The LLM provider ID ('openai', 'anthropic', 'google', etc.)
        prog (gr.Progress, optional): Gradio progress indicator
        
    Returns:
        gr.Dropdown: Updated dropdown component with fetched models
    """
    # First check if we should use the dynamic model fetcher
    dynamic_providers = ["openai", "anthropic", "google"]
    
    if llm_provider in dynamic_providers:
        try:
            from src.utils.llm_model_registry import get_models
            
            # Show progress indicator
            prog(0, desc=f"Fetching models for {llm_provider}...")
            
            # Fetch models from the provider
            models = await get_models(llm_provider)
            prog(1)  # Complete progress
            
            if models and len(models) > 0:
                # Return dropdown with dynamically fetched models
                return gr.Dropdown(
                    choices=models,
                    value=models[0] if models else None,
                    interactive=True,
                    allow_custom_value=True,
                )
            else:
                # Fallback to predefined models if dynamic fetching failed
                if llm_provider in config.model_names:
                    return gr.Dropdown(
                        choices=config.model_names[llm_provider],
                        value=config.model_names[llm_provider][0],
                        interactive=True,
                        allow_custom_value=True,
                    )
        except Exception as e:
            # Log the error and fall back to predefined models
            logger.error(f"Error fetching models for {llm_provider}: {e}")
            gr.Warning(f"Could not fetch models for {llm_provider}. Using predefined list instead.")
    
    # Use predefined models for providers not supporting dynamic fetching
    # or as fallback when dynamic fetching fails
    if llm_provider in config.model_names:
        return gr.Dropdown(
            choices=config.model_names[llm_provider],
            value=config.model_names[llm_provider][0],
            interactive=True,
            allow_custom_value=True,
        )
    else:
        return gr.Dropdown(
            choices=[],
            value="",
            interactive=True,
            allow_custom_value=True,
        )




async def update_mcp_server(mcp_file: str, webui_manager: WebuiManager):
    """
    Update the MCP server and save configuration for persistence.
    """
    if hasattr(webui_manager, "bu_controller") and webui_manager.bu_controller:
        logger.warning("⚠️ Close controller because mcp file has changed!")
        await webui_manager.bu_controller.close_mcp_client()
        webui_manager.bu_controller = None

    if not mcp_file or not os.path.exists(mcp_file) or not mcp_file.endswith('.json'):
        logger.warning(f"{mcp_file} is not a valid MCP file.")
        return None, gr.update(visible=False), None

    try:
        with open(mcp_file, 'r') as f:
            mcp_server = json.load(f)
            
        # Save configuration for persistence
        from src.utils.mcp_persistence import save_mcp_config
        save_mcp_config(mcp_server)
        
        # Return the configuration and update status
        return json.dumps(mcp_server, indent=2), gr.update(visible=True), mcp_server
    except Exception as e:
        logger.error(f"Error processing MCP file: {e}")
        return None, gr.update(visible=False), None


async def refresh_mcp_status(webui_manager: WebuiManager):
    """
    Refresh the MCP server status display.
    
    Args:
        webui_manager: The WebUI manager instance
        
    Returns:
        str: HTML content for the status display
    """
    if not hasattr(webui_manager, "bu_controller") or not webui_manager.bu_controller or not webui_manager.bu_controller.mcp_client:
        return "<div class='mcp-status-container'><p>No MCP servers active</p></div>"
    
    try:
        from src.utils.mcp_persistence import get_active_servers
        servers = get_active_servers(webui_manager.bu_controller.mcp_client)
        
        if not servers:
            return "<div class='mcp-status-container'><p>No MCP servers active</p></div>"
        
        html = "<div class='mcp-status-container'>"
        html += "<h3>MCP Servers Status</h3>"
        html += "<table class='mcp-status-table'>"
        html += "<tr><th>Server</th><th>Status</th><th>Available Tools</th></tr>"
        
        for server in servers:
            status_color = "green" if server["status"] == "Active" else "red"
            status_icon = "✅" if server["status"] == "Active" else "❌"
            
            html += f"<tr>"
            html += f"<td>{server['name']}</td>"
            html += f"<td><span style='color: {status_color};'>{status_icon} {server['status']}</span></td>"
            html += f"<td>{', '.join(server['tools']) if server['tools'] else 'No tools available'}</td>"
            html += f"</tr>"
        
        html += "</table></div>"
        
        # Add some CSS styling
        html += """
        <style>
        .mcp-status-container {
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
            background-color: rgba(0, 0, 0, 0.05);
        }
        .mcp-status-table {
            width: 100%;
            border-collapse: collapse;
        }
        .mcp-status-table th, .mcp-status-table td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        </style>
        """
        
        return html
    except Exception as e:
        logger.error(f"Error refreshing MCP status: {e}")
        return f"<div class='mcp-status-container'><p>Error refreshing MCP status: {str(e)}</p></div>"


def create_agent_settings_tab(webui_manager: WebuiManager):
    """
    Creates an agent settings tab.
    """
    input_components = set(webui_manager.get_components())
    tab_components = {}

    with gr.Group():
        with gr.Column():
            override_system_prompt = gr.Textbox(label="Override system prompt", lines=4, interactive=True)
            extend_system_prompt = gr.Textbox(label="Extend system prompt", lines=4, interactive=True)

    with gr.Group():
        with gr.Row():
            mcp_json_file = gr.File(label="MCP server json", interactive=True, file_types=[".json"])
            refresh_status_btn = gr.Button("Refresh Status", variant="secondary")
        
        mcp_server_config = gr.Textbox(label="MCP server", lines=6, interactive=True, visible=False)
        mcp_status_html = gr.HTML(label="MCP Status", value="<div class='mcp-status-container'><p>No MCP servers active</p></div>")

    with gr.Group():
        with gr.Row():
            llm_provider = gr.Dropdown(
                choices=[provider for provider, model in config.model_names.items()],
                label="LLM Provider",
                value="openai",
                info="Select LLM provider for LLM",
                interactive=True
            )
            llm_model_name = gr.Dropdown(
                label="LLM Model Name",
                choices=config.model_names['openai'],
                value="gpt-4o",
                interactive=True,
                allow_custom_value=True,
                info="Select a model in the dropdown options or directly type a custom model name"
            )
        with gr.Row():
            llm_temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.6,
                step=0.1,
                label="LLM Temperature",
                info="Controls randomness in model outputs",
                interactive=True
            )

            use_vision = gr.Checkbox(
                label="Use Vision",
                value=True,
                info="Enable Vision(Input highlighted screenshot into LLM)",
                interactive=True
            )

            ollama_num_ctx = gr.Slider(
                minimum=2 ** 8,
                maximum=2 ** 16,
                value=16000,
                step=1,
                label="Ollama Context Length",
                info="Controls max context length model needs to handle (less = faster)",
                visible=False,
                interactive=True
            )

        with gr.Row():
            llm_base_url = gr.Textbox(
                label="Base URL",
                value="",
                info="API endpoint URL (if required)"
            )
            llm_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value="",
                info="Your API key (leave blank to use .env)"
            )

    with gr.Group():
        with gr.Row():
            planner_llm_provider = gr.Dropdown(
                choices=[provider for provider, model in config.model_names.items()],
                label="Planner LLM Provider",
                info="Select LLM provider for LLM",
                value=None,
                interactive=True
            )
            planner_llm_model_name = gr.Dropdown(
                label="Planner LLM Model Name",
                interactive=True,
                allow_custom_value=True,
                info="Select a model in the dropdown options or directly type a custom model name"
            )
        with gr.Row():
            planner_llm_temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.6,
                step=0.1,
                label="Planner LLM Temperature",
                info="Controls randomness in model outputs",
                interactive=True
            )

            planner_use_vision = gr.Checkbox(
                label="Use Vision(Planner LLM)",
                value=False,
                info="Enable Vision(Input highlighted screenshot into LLM)",
                interactive=True
            )

            planner_ollama_num_ctx = gr.Slider(
                minimum=2 ** 8,
                maximum=2 ** 16,
                value=16000,
                step=1,
                label="Ollama Context Length",
                info="Controls max context length model needs to handle (less = faster)",
                visible=False,
                interactive=True
            )

        with gr.Row():
            planner_llm_base_url = gr.Textbox(
                label="Base URL",
                value="",
                info="API endpoint URL (if required)"
            )
            planner_llm_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value="",
                info="Your API key (leave blank to use .env)"
            )

    with gr.Row():
        max_steps = gr.Slider(
            minimum=1,
            maximum=1000,
            value=100,
            step=1,
            label="Max Run Steps",
            info="Maximum number of steps the agent will take",
            interactive=True
        )
        max_actions = gr.Slider(
            minimum=1,
            maximum=100,
            value=10,
            step=1,
            label="Max Number of Actions",
            info="Maximum number of actions the agent will take per step",
            interactive=True
        )

    with gr.Row():
        max_input_tokens = gr.Number(
            label="Max Input Tokens",
            value=128000,
            precision=0,
            interactive=True
        )
        tool_calling_method = gr.Dropdown(
            label="Tool Calling Method",
            value="auto",
            interactive=True,
            allow_custom_value=True,
            choices=["auto", "json_schema", "function_calling", "None"],
            visible=True
        )
    tab_components.update(dict(
        override_system_prompt=override_system_prompt,
        extend_system_prompt=extend_system_prompt,
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        llm_temperature=llm_temperature,
        use_vision=use_vision,
        ollama_num_ctx=ollama_num_ctx,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        planner_llm_provider=planner_llm_provider,
        planner_llm_model_name=planner_llm_model_name,
        planner_llm_temperature=planner_llm_temperature,
        planner_use_vision=planner_use_vision,
        planner_ollama_num_ctx=planner_ollama_num_ctx,
        planner_llm_base_url=planner_llm_base_url,
        planner_llm_api_key=planner_llm_api_key,
        max_steps=max_steps,
        max_actions=max_actions,
        max_input_tokens=max_input_tokens,
        tool_calling_method=tool_calling_method,
        mcp_json_file=mcp_json_file,
        mcp_server_config=mcp_server_config,
    ))
    webui_manager.add_components("agent_settings", tab_components)

    llm_provider.change(
        fn=lambda x: gr.update(visible=x == "ollama"),
        inputs=llm_provider,
        outputs=ollama_num_ctx
    )
    llm_provider.change(
        fn=update_model_dropdown,
        inputs=[llm_provider],
        outputs=[llm_model_name]
    )
    planner_llm_provider.change(
        fn=lambda x: gr.update(visible=x == "ollama"),
        inputs=[planner_llm_provider],
        outputs=[planner_ollama_num_ctx]
    )
    planner_llm_provider.change(
        fn=update_model_dropdown,
        inputs=[planner_llm_provider],
        outputs=[planner_llm_model_name]
    )

    async def update_wrapper(mcp_file):
        """Wrapper for handling MCP file updates."""
        mcp_config, visibility_update, mcp_server = await update_mcp_server(mcp_file, webui_manager)
        status_html = await refresh_mcp_status(webui_manager)
        yield mcp_config, visibility_update, status_html

    async def refresh_status_wrapper():
        """Wrapper for refreshing MCP status."""
        status_html = await refresh_mcp_status(webui_manager)
        return status_html

    # Handle MCP file uploads
    mcp_json_file.change(
        update_wrapper,
        inputs=[mcp_json_file],
        outputs=[mcp_server_config, mcp_server_config, mcp_status_html]
    )
    
    # Handle status refresh button
    refresh_status_btn.click(
        refresh_status_wrapper,
        outputs=[mcp_status_html]
    )
    
    # Load saved MCP configuration on startup
    async def load_saved_config():
        """Load saved MCP configuration on startup."""
        try:
            from src.utils.mcp_persistence import load_mcp_config
            saved_config = load_mcp_config()
            
            if saved_config:
                logger.info("Loading saved MCP configuration")
                # Update the UI with saved configuration
                mcp_config = json.dumps(saved_config, indent=2)
                
                # Initialize MCP client with saved configuration
                if hasattr(webui_manager, "bu_controller") and webui_manager.bu_controller:
                    webui_manager.bu_controller.mcp_server_config = saved_config
                    webui_manager.bu_controller.mcp_client = await setup_mcp_client_and_tools(saved_config)
                
                # Refresh status display
                status_html = await refresh_mcp_status(webui_manager)
                
                return mcp_config, gr.update(visible=True), status_html
            
        except Exception as e:
            logger.error(f"Error loading saved MCP configuration: {e}")
        
        return None, gr.update(visible=False), "<div class='mcp-status-container'><p>No MCP servers active</p></div>"
    
    # Schedule loading of saved configuration
    demo = gr.Blocks.current_block
    if demo:
        demo.load(load_saved_config, outputs=[mcp_server_config, mcp_server_config, mcp_status_html])
    else:
        logger.warning("Could not schedule loading of saved MCP configuration: no active Gradio block")

