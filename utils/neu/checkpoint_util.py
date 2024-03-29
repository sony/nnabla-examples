# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import queue

import nnabla as nn
from nnabla.logger import logger

global prev_save_paths
prev_save_paths = queue.Queue()


def save_checkpoint(path, current_iter, solvers, n_keeps=-1, split_h5_per_solver=False):
    """Saves the checkpoint file which contains the params and its state info.

        Args:
            path: Path to the directory the checkpoint file is stored in.
            current_iter: Current iteretion of the training loop.
            solvers: A dictionary about solver's info, which is like;
                     solvers = {"identifier_for_solver_0": solver_0,
                               {"identifier_for_solver_1": solver_1, ...}
                     The keys are used just for state's filenames, so can be anything.
                     Also, you can give a solver object if only one solver exists.
                     Then, the "" is used as an identifier.
            n_keeps: Number of latest checkpoints to keep. If -1, all checkpoints are kept.
                     Note that we assume save_checkpoint is called from a single line in your script.
                     When you have to call this from multiple lines, n_keeps must be -1 (you have to disable n_keeps).
            split_h5_per_solver: If True, save several h5 files for parameters.
                                 Each h5 file contains subset of parameters that each solver has.

        Examples:
            # Create computation graph with parameters.
            pred = construct_pred_net(input_Variable, ...)

            # Create solver and set parameters.
            solver = S.Adam(learning_rate)
            solver.set_parameters(nn.get_parameters())

            # If you have another_solver like,
            # another_solver = S.Sgd(learning_rate)
            # another_solver.set_parameters(nn.get_parameters())

            # Training loop.
            for i in range(start_point, max_iter):
                pred.forward()
                pred.backward()
                solver.zero_grad()
                solver.update()
                save_checkpoint(path, i, solver)

                # If you have another_solver,
                # save_checkpoint(path, i,
                      {"solver": solver, "another_solver": another})

        Notes:
            It generates the checkpoint file (.json) which is like;
            checkpoint_1000 = {
                    "":{
                        "states_path": <path to the states file>
                        "params_names":["conv1/conv/W", ...],
                        "num_update":1000
                       },
                    "current_iter": 1000
                    }

            If you have multiple solvers.
            checkpoint_1000 = {
                    "generator":{
                        "states_path": <path to the states file>,
                        "params_names":["deconv1/conv/W", ...],
                        "num_update":1000
                       },
                    "discriminator":{
                        "states_path": <path to the states file>,
                        "params_names":["conv1/conv/W", ...],
                        "num_update":1000
                       },
                    "current_iter": 1000
                    }

    """

    if isinstance(solvers, nn.solver.Solver):
        solvers = {"": solvers}

    assert isinstance(solvers, dict), \
        "`solvers` must be either Solver object or dict of { `name`: Solver }."

    checkpoint_info = dict()
    save_paths = []

    for solvername, solver_obj in solvers.items():
        prefix = "{}_".format(solvername.replace(
            "/", "_")) if solvername else ""
        partial_info = dict()

        # save solver states.
        states_fname = prefix + 'states_{}.h5'.format(current_iter)
        states_path = os.path.join(path, states_fname)
        solver_obj.save_states(states_path)
        save_paths.append(states_path)
        # save relative path to support moving a saved directory
        partial_info["states_path"] = states_fname

        # save registered parameters' name. (just in case)
        params_names = [k for k in solver_obj.get_parameters().keys()]
        partial_info["params_names"] = params_names

        # save the number of solver update.
        num_update = getattr(solver_obj.get_states()[params_names[0]], "t")
        partial_info["num_update"] = num_update

        # save parameters per solver
        if split_h5_per_solver:
            solver_params_fname = f"{solvername}_params_{current_iter}.h5"
            solver_params_path = os.path.join(path, solver_params_fname)
            nn.save_parameters(path=solver_params_path,
                               params=solver_obj.get_parameters())
            save_paths.append(solver_params_path)

            # save relative path so to support moving a saved directory
            partial_info["params_path"] = solver_params_fname

        checkpoint_info[solvername] = partial_info

    # save parameters.
    if not split_h5_per_solver:
        params_fname = 'params_{}.h5'.format(current_iter)
        params_path = os.path.join(path, params_fname)
        nn.parameter.save_parameters(params_path)
        save_paths.append(params_path)

        # save relative path so to support moving a saved directory
        checkpoint_info["params_path"] = params_fname

    checkpoint_info["current_iter"] = current_iter

    # save checkpoint
    checkpoint_fname = 'checkpoint_{}.json'.format(current_iter)
    filename = os.path.join(path, checkpoint_fname)

    with open(filename, 'w') as f:
        json.dump(checkpoint_info, f)

    logger.info("Checkpoint save (.json): {}".format(filename))
    save_paths.append(filename)

    # keep only n_keeps latest checkpoints.
    if n_keeps > 0:
        global prev_save_paths
        prev_save_paths.put(save_paths)
        if prev_save_paths.qsize() > n_keeps:
            oldest = prev_save_paths.get()
            for path in oldest:
                os.remove(path)

    return


def _get_full_path(path, base_path):
    # for backward compatibility

    # case1: path is absolute
    if os.path.exists(path):
        return path

    # case2: path is reative based on base_path
    path1 = os.path.join(base_path, path)
    if os.path.exists(path1):
        return path1

    # otherwise: raise
    raise ValueError(
        f"Given path doesn't exist. (path: {path}, base_path: {base_path})")


def load_checkpoint(path, solvers):
    """Given the checkpoint file, loads the parameters and solver states.

        Args:
            path: Path to the checkpoint file.
            solvers: A dictionary about solver's info, which is like;
                     solvers = {"identifier_for_solver_0": solver_0,
                               {"identifier_for_solver_1": solver_1, ...}
                     The keys are used for retrieving proper info from the checkpoint.
                     so must be the same as the one used when saved.
                     Also, you can give a solver object if only one solver exists.
                     Then, the "" is used as an identifier.

        Returns:
            current_iter: The number of iteretions that the training resumes from.
                          Note that this assumes that the numbers of the update for
                          each solvers is the same.

        Examples:
            # Create computation graph with parameters.
            pred = construct_pred_net(input_Variable, ...)

            # Create solver and set parameters.
            solver = S.Adam(learning_rate)
            solver.set_parameters(nn.get_parameters())

            # AFTER setting parameters.
            start_point = load_checkpoint(path, solver)

            # Training loop.

        Notes:
            It requires the checkpoint file. For details, refer to save_checkpoint;
            checkpoint_1000 = {
                    "":{
                        "states_path": <path to the states file>
                        "params_names":["conv1/conv/W", ...],
                        "num_update":1000
                       },
                    "current_iter": 1000
                    }

            If you have multiple solvers.
            checkpoint_1000 = {
                    "generator":{
                        "states_path": <path to the states file>,
                        "params_names":["deconv1/conv/W", ...],
                        "num_update":1000
                       },
                    "discriminator":{
                        "states_path": <path to the states file>,
                        "params_names":["conv1/conv/W", ...],
                        "num_update":1000
                       },
                    "current_iter": 1000
                    }

    """

    if isinstance(solvers, nn.solver.Solver):
        solvers = {"": solvers}
    assert isinstance(solvers, dict), \
        "`solvers` must be either Solver object or dict of { `name`: Solver }."

    assert os.path.isfile(path), "checkpoint file not found"
    base_path = os.path.dirname(path)

    # load checkpoint
    with open(path, 'r') as f:
        checkpoint_info = json.load(f)
    logger.info("Checkpoint load (.json): {}".format(path))

    # load parameters (stored in global).
    if "params_path" in checkpoint_info:
        params_path = _get_full_path(checkpoint_info["params_path"], base_path)
        assert os.path.isfile(params_path), "parameters file not found."

        nn.parameter.load_parameters(params_path)

    for solvername, solver_obj in solvers.items():
        partial_info = checkpoint_info[solvername]
        if set(solver_obj.get_parameters().keys()) != set(partial_info["params_names"]):
            logger.warning("Detected parameters do not match.")

        # load solver states.
        states_path = _get_full_path(partial_info["states_path"], base_path)
        assert os.path.isfile(states_path), "states file not found."

        # set solver states.
        if solvername == "ema":
            try:
                solver_obj.load_states(states_path)
            except:
                logger.info("load state for ema is failed.")
        else:
            solver_obj.load_states(states_path)

        # load parameters belonging to this solver if exists
        if "params_path" in partial_info:
            solver_params_path = _get_full_path(
                partial_info["params_path"], base_path)
            nn.load_parameters(solver_params_path)

    # get current iteration. note that this might differ from the numbers of update.
    current_iter = checkpoint_info["current_iter"]

    return current_iter
