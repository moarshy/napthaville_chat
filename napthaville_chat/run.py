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
    init_persona_info = json.loads(init_persona_info)
    # target_persona_info: {"name": "Maria Lopez", "act_description": "sleeping"}

    target_persona_info = await target_persona_info_task(
        task='get_personal_info', 
        task_params={
            'persona_name': target_persona_name,
        }
    )
    target_persona_info = json.loads(target_persona_info)

    logger.info(f"init_persona_info: {init_persona_info}")
    logger.info(f"target_persona_info: {target_persona_info}")
    logger.info(f"Type of init_persona_info: {type(init_persona_info)}")

    init_persona_chat_params = {
        'init_persona_name': init_persona_name,
        'target_persona_name': target_persona_info['name'],
        'target_persona_description': target_persona_info['act_description'],
    }

    target_persona_chat_params = {
        'init_persona_name': target_persona_name,
        'target_persona_name': init_persona_info['name'],
        'target_persona_description': init_persona_info['act_description'],
    }

    curr_chat = []
    for i in range(8):
        init_persona_chat_params['curr_chat'] = json.dumps(curr_chat)
        init_utterance = await init_persona_info_task(
            task='get_utterence',
            task_params=init_persona_chat_params,
        )
        logger.info(f"init_utterance: {init_utterance}")

        target_persona_chat_params['curr_chat'] = json.dumps(json.loads(init_utterance)['curr_chat'])
        target_utterance = await target_persona_info_task(
            task='get_utterence',
            task_params=target_persona_chat_params,
        )
        logger.info(f"target_utterance: {target_utterance}")

        curr_chat = json.loads(target_utterance)['curr_chat']
        logger.info(f"curr_chat: {curr_chat}")

    return json.dumps(curr_chat)
