import logging
import json
from typing import Dict
from napthaville_chat.schemas import InputSchema
from naptha_sdk.task import Task as NapthaTask


logger = logging.getLogger(__name__)


async def run(inputs: InputSchema, worker_nodes = None, orchestrator_node = None, flow_run = None, cfg: Dict = None):
    logger.info(f"Running with inputs: {inputs}")
    logger.info(f"Worker nodes: {worker_nodes}")
    logger.info(f"Orchestrator node: {orchestrator_node}")

    if len(worker_nodes) < 2:
        init_persona_node = worker_nodes[0]
        target_persona_node = worker_nodes[0]

    init_persona_node = worker_nodes[0]
    target_persona_node = worker_nodes[1]

    init_persona_name = inputs.init_persona
    target_persona_name = inputs.target_persona

    init_persona_info_task = NapthaTask(
        name = 'get_personal_info',
        fn = 'napthaville_module',
        worker_node = init_persona_node,
        orchestrator_node = orchestrator_node,
        flow_run = flow_run,
    )

    target_persona_info_task = NapthaTask(
        name = 'get_personal_info',
        fn = 'napthaville_module',
        worker_node = target_persona_node,
        orchestrator_node = orchestrator_node,
        flow_run = flow_run,
    )

    logger.info(f"Running init_persona_info_task with inputs: {init_persona_name}")
    init_persona_info = await init_persona_info_task(
        task='get_personal_info', 
        task_params={
            'persona_name': init_persona_name,
        }
    )

    target_persona_info = await target_persona_info_task(
        task='get_personal_info', 
        task_params={
            'persona_name': target_persona_name,
        }
    )

    logger.info(f"init_persona_info: {init_persona_info}")
    logger.info(f"target_persona_info: {target_persona_info}")
    logger.info(f"Type of init_persona_info: {type(init_persona_info)}")

    # init_persona_chat_params = {
    #     'init_persona': init_persona_name,
    #     'target_persona': target_persona_name,
    #     'maze_folder': inputs.maze_folder,
    # }

    # target_persona_chat_params = {
    #     'init_persona': target_persona_name,
    #     'target_persona': init_persona_name,
    #     'maze_folder': inputs.maze_folder,
    # }

    # init_persona_chat_task = ...
    # target_persona_chat_task = ...

    # curr_chat = []
    # for i in range(8):

    results = {
        'init_persona_info': init_persona_info,
        'target_persona_info': target_persona_info,
    }

    return json.dumps(results)
