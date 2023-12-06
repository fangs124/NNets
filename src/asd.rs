CS::Train => {
                let mut stdout = stdout();
                let mut stderr = stderr();
                let mut stream: BufWriter<&std::io::Stdout> = BufWriter::new(&stdout);
                let mut stream_err: BufWriter<&std::io::Stderr> = BufWriter::new(&stderr);
                writeln!(stream, "Press q to stop.")?;
                let start_time = Instant::now();
                let mut now = Instant::now();
                let mut prev_w = 0.0;
                let mut prev_l = 0.0;
                let mut prev_d = 0.0;

                /* various counters for data gathering */
                let mut loop_counter: usize = 0;
                let mut hundykay_counter: usize = 0;
                let mut invalid_total: usize = 0;
                let mut invalid_counter: usize = 0;
                let mut self_play: usize = 0;
                let mut x_trained: usize = 0;
                let mut o_trained: usize = 0;
                let mut network_wins: usize = 0;
                let mut random_wins: usize = 0;
                let mut draws: usize = 0;
                let mut self_draws: usize = 0;

                /* training flow controls */
                let mut is_training = true;
                let mut is_playing_self = false;
                let mut train_x: bool = true;

                /* logging first thousand games or so */
                let mut f = File::create(format!("{:?}log.txt", node_count.clone())).unwrap();
                writeln!(&mut f, "First 1 second:")?;
                writeln!(
                    stream,
                    "N wins: {}, R wins: {}, Draws: {}, [{:.2}:{:.2}:{:.2}]",
                    network_wins, random_wins, draws, 0, 0, 0
                )?;
                writeln!(
                    stream,
                    "Time Training: {:.2?}, Trained: {}k, (X:{},O:{}), Self Play: {}",
                    start_time.elapsed(),
                    (x_trained + o_trained + self_play) / 1000,
                    x_trained,
                    o_trained,
                    self_play
                )?;
                stream.flush()?;

                while is_training {
                    #[allow(non_snake_case)]
                    let mut vec_dCda_x: Vec<Vec<f64>> = Vec::new();
                    #[allow(non_snake_case)]
                    let mut vec_dCda_o: Vec<Vec<f64>> = Vec::new();

                    let return_val = unsafe { GetAsyncKeyState(0x51 as i32) };
                    if return_val & 0x01 != 0 {
                        //stop training
                        choice = CS::NoChoice;
                        is_training = false;
                        continue;
                    };

                    gb.new_game();
                    /*
                    if x_turn {
                        network.set_input(gb.to_vec_bool_x());
                    } else {
                        network.set_input(gb.to_vec_bool_o());
                    }*/
                    //let vec_bool = gb.to_vec_bool();

                    let mut x_turn: bool = true;
                    let mut x_moves: Vec<usize> = Vec::new();
                    let mut o_moves: Vec<usize> = Vec::new();
                    let mut x_states: Vec<Vec<bool>> = Vec::new();
                    let mut o_states: Vec<Vec<bool>> = Vec::new();

                    while gb.game_state() == GS::Ongoing {
                        let mut vec_bool = Vec::new();
                        if x_turn {
                            vec_bool.append(&mut gb.to_vec_bool_x().clone().to_owned());
                        } else {
                            vec_bool.append(&mut gb.to_vec_bool_o().clone().to_owned());
                        }
                        //let vec_bool = gb.to_vec_bool();

                        if x_turn {
                            x_states.push(vec_bool.clone())
                        } else {
                            o_states.push(vec_bool.clone())
                        }

                        // computer's turn
                        if (x_turn == train_x) || is_playing_self {
                            let mut invalid_moves: Vec<usize> = Vec::new();
                            let mut invalid_states: Vec<Vec<bool>> = Vec::new();
                            let output: Vec<f64> = network.forward_prop(vec_bool.clone());
                            let invalid_indices: Vec<usize> = output
                                .iter()
                                .copied()
                                .enumerate()
                                .filter(|&(_, x)| x >= 0.01)
                                .filter(|&(i, _)| !gb.is_valid_move(&BB::MOVES[i]))
                                .map(|(index, _)| index)
                                .collect();
                            invalid_counter += invalid_indices.len();
                            invalid_total += invalid_indices.len();
                            let mut dCda_invalids = vec![0.0; BB::MOVES.len()];
                            for i in invalid_indices.clone() {
                                dCda_invalids[i] = (-0.75) * invalid_indices.clone().len() as f64;
                                invalid_states.push(vec_bool.clone());
                                invalid_moves.push(i);
                            }
                            let mut vec_dCda_invalids: Vec<Vec<f64>> = Vec::new();
                            for i in invalid_indices {
                                vec_dCda_invalids.push(dCda_invalids.clone())
                            }
                            let vec: Vec<bool> = Vec::new();
                            let vec: Vec<(Vec<bool>, Vec<f64>)> = invalid_states
                                .iter()
                                .cloned()
                                .zip(vec_dCda_invalids)
                                .collect();
                            network.train(vec, 2, invalid_moves);

                            if let Some(index) = output
                                .iter()
                                .copied()
                                .enumerate()
                                .filter(|&(i, _)| gb.is_valid_move(&BB::MOVES[i]))
                                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                .map(|(index, _)| index)
                            {
                                match gb.make_move(BB::MOVES[index]) {
                                    Err(err) => writeln!(stream_err, "{:?}", err)?,
                                    _ => {}
                                }
                                let len = BB::MOVES.len();
                                #[allow(non_snake_case)]
                                let mut dCda = vec![0.0; len];
                                dCda[index] = 1.0;
                                if x_turn {
                                    x_moves.push(index);
                                    vec_dCda_x.push(dCda);
                                } else {
                                    o_moves.push(index);
                                    vec_dCda_o.push(dCda);
                                }
                            } else {
                                gb.print_gameboard();
                                let bb: BitBoard = gb.x_b | gb.o_b;
                                writeln!(stream_err, "bb: {:#?}", bb)?;
                                writeln!(
                                    stream_err,
                                    "output: {:#?}",
                                    output.clone().iter().copied().enumerate()
                                )?;
                                panic!("network-output error!"); //todo: error handling
                            }
                        }
                        //random's turn
                        else {
                            let indices: Vec<usize> = (0..9)
                                .collect::<Vec<usize>>()
                                .iter()
                                .filter(|&i| gb.is_valid_move(&BB::MOVES[*i]))
                                .map(|&x| x)
                                .collect();
                            let rand_num = rng.gen_range(0..indices.len());
                            gb.make_move(BB::MOVES[indices[rand_num]])
                                .expect("invalid moves are impossible!");
                            let len = BB::MOVES.len();
                            #[allow(non_snake_case)]
                            let mut dCda = vec![0.0; len];
                            dCda[indices[rand_num]] = 1.0;
                            if x_turn {
                                x_moves.push(indices[rand_num]);
                                vec_dCda_x.push(dCda);
                            } else {
                                o_moves.push(indices[rand_num]);
                                vec_dCda_o.push(dCda);
                            }
                        }
                        x_turn = !x_turn
                    }

                    // game ended
                    match gb.game_state() {
                        GS::XWin => {
                            #[allow(non_snake_case)]
                            for dCda_o in vec_dCda_o.iter_mut() {
                                *dCda_o = dCda_o.iter().map(|x| *x * -1.0).collect();
                            }
                            if !is_playing_self {
                                if train_x {
                                    //network wins
                                    network_wins += 1;

                                    //network won as x
                                    let vec: Vec<(Vec<bool>, Vec<f64>)> =
                                        x_states.iter().cloned().zip(vec_dCda_x).collect();
                                    network.train(vec, 2, x_moves);

                                    //random lost as o

                                    let vec: Vec<(Vec<bool>, Vec<f64>)> =
                                        o_states.iter().cloned().zip(vec_dCda_o).collect();
                                    network.train(vec, 2, o_moves);

                                    //network.update();
                                    x_trained += 1;
                                } else {
                                    //network lost
                                    random_wins += 1;

                                    //random won as x

                                    let vec: Vec<(Vec<bool>, Vec<f64>)> =
                                        x_states.iter().cloned().zip(vec_dCda_x).collect();
                                    network.train(vec, 2, x_moves);

                                    //network lost as o
                                    let vec: Vec<(Vec<bool>, Vec<f64>)> =
                                        o_states.iter().cloned().zip(vec_dCda_o).collect();
                                    network.train(vec, 2, o_moves);
                                    //network.update();
                                    o_trained += 1;
                                }
                            } else {
                                //playing self
                                #[allow(non_snake_case)]
                                let vec: Vec<(
                                    Vec<bool>,
                                    Vec<f64>,
                                )> = x_states.iter().cloned().zip(vec_dCda_x).collect();
                                network.train(vec, 2, x_moves);

                                let vec: Vec<(Vec<bool>, Vec<f64>)> =
                                    o_states.iter().cloned().zip(vec_dCda_o).collect();
                                network.train(vec, 2, o_moves);
                                //network.update();
                                self_play += 1;
                            }
                        }
                        GS::OWin => {
                            #[allow(non_snake_case)]
                            for dCda_x in vec_dCda_x.iter_mut() {
                                *dCda_x = dCda_x.iter().map(|x| *x * -1.0).collect();
                            }
                            if !is_playing_self {
                                if train_x {
                                    //network lost
                                    random_wins += 1;

                                    //network lost as x
                                    let vec: Vec<(Vec<bool>, Vec<f64>)> =
                                        x_states.iter().cloned().zip(vec_dCda_x).collect();
                                    network.train(vec, 2, x_moves);

                                    //random won as o

                                    let vec: Vec<(Vec<bool>, Vec<f64>)> =
                                        o_states.iter().cloned().zip(vec_dCda_o).collect();
                                    network.train(vec, 2, o_moves);

                                    //network.update();
                                    x_trained += 1;
                                } else {
                                    //network won
                                    network_wins += 1;
                                    /*
                                    #[allow(non_snake_case)]
                                    for dCda_x in vec_dCda_x.iter_mut() {
                                        *dCda_x = dCda_x.iter().map(|x| *x * -1.0).collect();
                                    }
                                    */

                                    //random lost as x

                                    let vec: Vec<(Vec<bool>, Vec<f64>)> =
                                        x_states.iter().cloned().zip(vec_dCda_x).collect();
                                    network.train(vec, 2, x_moves);

                                    //network won as o
                                    let vec: Vec<(Vec<bool>, Vec<f64>)> =
                                        o_states.iter().cloned().zip(vec_dCda_o).collect();
                                    network.train(vec, 2, o_moves);
                                    //network.update();
                                    o_trained += 1;
                                }
                            } else {
                                //playing self,
                                let vec: Vec<(Vec<bool>, Vec<f64>)> =
                                    x_states.iter().cloned().zip(vec_dCda_x).collect();
                                network.train(vec, 2, x_moves);

                                let vec: Vec<(Vec<bool>, Vec<f64>)> =
                                    o_states.iter().cloned().zip(vec_dCda_o).collect();
                                network.train(vec, 2, o_moves);
                                //network.update();
                                self_play += 1;
                            }
                        }
                        GS::Tie => {
                            if !is_playing_self {
                                draws += 1
                            } else {
                                self_play += 1;
                                self_draws += 1;
                                // here to train draws against self.
                            }
                        }
                        _ => panic!("game state should be impossible!"),
                    }
                    /*
                    if start_time.elapsed() <= Duration::from_secs(5) {
                        let total: usize = network_wins + random_wins + draws;
                        /*
                        println!("X wins: {}, O wins: {}, Draws: {} ({:.2}%), Time Training: {:.2?}, X trained: {}, O trained: {}",
                            x_wins, o_wins, draws, percent, start_time.elapsed(), x_trained, o_trained);
                        */
                        /*
                        print!("\rN wins: {}, R wins: {}, Draws: {}, [{:.2}:{:.2}:{:.2}]\nTime Training: {:.2?}, Trained: {}k, (X:{},O:{}), Self Play: {}",
                            network_wins, random_wins, draws,
                            (network_wins as f64)/(total as f64),(random_wins as f64)/(total as f64),(draws as f64)/(total as f64),
                            start_time.elapsed(), (x_trained+o_trained+self_play)/1000,x_trained, o_trained, self_play);
                        */
                        stream.execute(cursor::MoveUp(2)).unwrap();
                        stream
                            .execute(terminal::Clear(terminal::ClearType::FromCursorDown))
                            .unwrap();
                        writeln!(
                            stream,
                            "N wins: {}, R wins: {}, Draws: {}, [{:.2}:{:.2}:{:.2}]",
                            network_wins,
                            random_wins,
                            draws,
                            100.0 * (network_wins as f64) / (total as f64),
                            100.0 * (random_wins as f64) / (total as f64),
                            100.0 * (draws as f64) / (total as f64)
                        )?;
                        writeln!(
                            stream,
                            "Time Training: {:.2?}, Trained: {}k, (X:{},O:{}), Self Play: {} (Draws: {:.2}%)",
                            start_time.elapsed(),
                            (x_trained + o_trained) / 1000,
                            x_trained,
                            o_trained,
                            self_play, (self_draws as f64/ self_play as f64)*100.0
                        )?;
                        stream.flush()?;
                    }
                    */
                    if (loop_counter % 1000) == 0 {
                        //1k
                        network.update();
                    }
                    if (loop_counter % 100000) == 0 {
                        //100K
                        let total: usize = network_wins + random_wins + draws;

                        hundykay_counter += 1;
                        stream.execute(cursor::MoveUp(2)).unwrap();
                        stream
                            .execute(terminal::Clear(terminal::ClearType::FromCursorDown))
                            .unwrap();
                        writeln!(
                            stream,
                            "N wins: {}, R wins: {}, Draws: {}, [{:.2}+({:.2}):{:.2}+({:.2}):{:.2}+({:.2})] (100k-epoch: {}, {:.2?}) invalid this epoch: {}",
                            network_wins,
                            random_wins,
                            draws,
                            100.0 * (network_wins as f64) / (total as f64),
                            (100.0 * (network_wins as f64) / (total as f64)) - prev_w,
                            100.0 * (random_wins as f64) / (total as f64),
                            (100.0 * (random_wins as f64) / (total as f64)) - prev_l,
                            100.0 * (draws as f64) / (total as f64),
                            (100.0 * (draws as f64) / (total as f64)) - prev_d,
                            hundykay_counter,
                            now.elapsed(),
                            invalid_counter,
                        )?;
                        writeln!(
                            stream,
                            "Time Training: {:.2?}, Trained: {}k, (X:{},O:{}), Self Play: {} (Draws: {:.2}%) invalid_total: {}",
                            start_time.elapsed(),
                            (x_trained + o_trained) / 1000,
                            x_trained,
                            o_trained,
                            self_play, (self_draws as f64/ self_play as f64)*100.0,
                            invalid_total,
                        )?;
                        writeln!(
                            f,
                            "N wins: {}, R wins: {}, Draws: {}, [{:.2}:{:.2}:{:.2}] (100K-epoch: {}, {:.2?})",
                            network_wins,
                            random_wins,
                            draws,
                            100.0 * (network_wins as f64) / (total as f64),
                            100.0 * (random_wins as f64) / (total as f64),
                            100.0 * (draws as f64) / (total as f64),
                            hundykay_counter,
                            now.elapsed(),
                        )?;
                        writeln!(
                            f,
                            "Time Training: {:.2?}, Trained: {}k, (X:{},O:{}), Self Play: {} (Draws: {:.2}%)",
                            start_time.elapsed(),
                            (x_trained + o_trained) / 1000,
                            x_trained,
                            o_trained,
                            self_play, (self_draws as f64/ self_play as f64)*100.0
                        )?;
                        prev_w = 100.0 * (network_wins as f64) / (total as f64);
                        prev_l = 100.0 * (random_wins as f64) / (total as f64);
                        prev_d = 100.0 * (draws as f64) / (total as f64);
                        stream.flush()?;
                        loop_counter = 0;
                        self_play = 0;
                        x_trained = 0;
                        o_trained = 0;
                        network_wins = 0;
                        random_wins = 0;
                        draws = 0;
                        self_draws = 0;
                        invalid_counter = 0;
                        now = Instant::now();
                    }
                    loop_counter += 1;
                    train_x = rng.gen();
                    is_playing_self = rng.gen();
                }
            }