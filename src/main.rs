mod gameboards;
mod nnets;

use crate::gameboards::*;
use crate::nnets::*;

use crossterm::{cursor, terminal, ExecutableCommand, Result};
use inquire::Select;
use rand::Rng;
use std::env;
use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::fs::File;
use std::io::{stderr, stdout, BufReader, BufWriter, Read, Write};
use std::time::{Duration, Instant};
use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;

impl InputType for bool {
    fn to_f32(&self) -> f32 {
        match self {
            true => 1.0,
            false => 0.0,
        }
    }
}

impl BitBoard {
    pub fn to_vec_bool(&self) -> Vec<bool> {
        let mut vec: Vec<bool> = Vec::new();
        for i in 0..=11 {
            if i % 4 != 0 {
                vec.push(self.get_val() & (1 << (11 - i)) != 0);
            }
        }
        //println!("{:?}", self);
        //println!("{:?}", vec);
        vec
    }
}

impl GameBoard {
    pub fn to_vec_bool(&self) -> Vec<bool> {
        let mut vec: Vec<bool> = Vec::new();
        vec.append(&mut self.x_b.to_vec_bool());
        vec.append(&mut self.o_b.to_vec_bool());
        vec.push(self.t_b == BB::ONES);
        vec
    }
}

//Choice State
#[derive(PartialEq, Eq, Clone)]
enum CS {
    NoChoice,
    Play,
    FinishedGame,
    Train,
    Sample,
    Quit,
}

impl Display for CS {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CS::NoChoice => f.write_str("NoChoice"),
            CS::Play => f.write_str("Play"),
            CS::Train => f.write_str("Train"),
            CS::FinishedGame => f.write_str("FinishedGame"),
            CS::Sample => f.write_str("Sample"),
            CS::Quit => f.write_str("Quit"),
        }
    }
}
fn main() -> std::io::Result<()> {
    //println!("{:?}", BB::OUT_OF_BOUNDS.to_vec_bool());
    env::set_var("RUST_BACKTRACE", "1");
    let mut choice: CS = CS::NoChoice;
    let start_q = String::from("YiffYaffYoff v0.1:");
    let start_opt: Vec<CS> = vec![
        CS::Play,
        CS::Train,
        CS::Sample,
        CS::Quit,
        /*"Sample Game,  Save NNet, Load NNet, Print NNet,*/
    ];

    //deal with loading/generating network here
    let network_q = String::from("New Network?");
    let network_opt = vec!["New Network", "Load Network"];
    let node_count: Vec<usize> = vec![6, 9]; //param_here
    let mut gb = GameBoard::NEW_GAMEBOARD;
    let mut network: Network<bool>;
    let mut rng = rand::thread_rng();
    match Select::new(&network_q, network_opt)
        .prompt()
        .expect("Prompt error")
    {
        "New Network" => {
            let now = Instant::now();
            println!("Initializing...");
            network = Network::new_default(node_count, gb.to_vec_bool());
            println!("Elapsed: {:.2?}", now.elapsed());
        }
        "Load Network" => {
            let now = Instant::now();
            println!("Loading...");
            let file = File::open("network.json")?;
            let mut buf_reader = BufReader::new(file);
            let mut contents = String::new();
            buf_reader.read_to_string(&mut contents)?;
            network = serde_json::from_str(&contents).unwrap();
            println!("Elapsed: {:.2?}", now.elapsed());
        }
        _ => {
            panic!("match error!");
        }
    }

    while choice != CS::Quit {
        match choice {
            CS::NoChoice => {
                choice = Select::new(&start_q, start_opt.clone())
                    .prompt()
                    .expect("Prompt error")
            }
            CS::Quit => {
                let quit_q = "Save Network?";
                let quit_opt = vec!["Yes", "No"];
                match Select::new(&quit_q, quit_opt)
                    .prompt()
                    .expect("Prompt error")
                {
                    "Yes" => {
                        // save network
                        let file = File::create("network.json")?;

                        serde_json::to_writer(file, &network)?;
                    }
                    "No" => {}
                    _ => panic!("Prompt error"),
                }
            }
            CS::Train => {
                let mut stdout = stdout();
                let mut stderr = stderr();
                let mut stream: BufWriter<&std::io::Stdout> = BufWriter::new(&stdout);
                let mut stream_err: BufWriter<&std::io::Stderr> = BufWriter::new(&stderr);
                writeln!(stream, "Press q to stop.")?;
                let mut now = Instant::now();
                let start_time = Instant::now();

                /* various counters for data gathering */
                let mut self_play: u128 = 0;
                let mut x_trained: u128 = 0;
                let mut o_trained: u128 = 0;
                let mut network_wins: u128 = 0;
                let mut random_wins: u128 = 0;
                let mut draws: u128 = 0;

                /* training flow controls */
                let mut is_training = true;
                let mut is_playing_self = false;
                let mut train_x: bool = true;

                /* logging first thousand games or so */
                let mut f = File::create("log.txt").unwrap();
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
                    let mut vec_dCda_x: Vec<Vec<f32>> = Vec::new();
                    #[allow(non_snake_case)]
                    let mut vec_dCda_o: Vec<Vec<f32>> = Vec::new();

                    let return_val = unsafe { GetAsyncKeyState(0x51 as i32) };
                    if return_val & 0x01 != 0 {
                        //stop training
                        choice = CS::NoChoice;
                        is_training = false;
                        continue;
                    };

                    gb.new_game();
                    network.set_input(gb.to_vec_bool());

                    let mut x_turn: bool = true;
                    let mut x_moves: Vec<usize> = Vec::new();
                    let mut o_moves: Vec<usize> = Vec::new();
                    let mut x_states: Vec<Vec<bool>> = Vec::new();
                    let mut o_states: Vec<Vec<bool>> = Vec::new();

                    while gb.game_state() == GS::Ongoing {
                        let vec_bool = gb.to_vec_bool();
                        //println!("{:?}", vec_bool);
                        if x_turn {
                            x_states.push(vec_bool.clone())
                        } else {
                            o_states.push(vec_bool.clone())
                        }

                        // computer's turn
                        if (x_turn == train_x) || is_playing_self {
                            let output: Vec<f32> = network.forward_prop(vec_bool.clone());
                            if let Some(index) = output
                                .iter()
                                .copied()
                                .enumerate()
                                .filter(|&(i, _)| gb.is_valid_move(&BB::POSSIBLE_MOVES[i]))
                                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                .map(|(index, _)| index)
                            {
                                match gb.make_move(BB::POSSIBLE_MOVES[index]) {
                                    Err(err) => writeln!(stream_err, "{:?}", err)?,
                                    _ => {}
                                }
                                let len = BB::POSSIBLE_MOVES.len();
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
                                .filter(|&i| gb.is_valid_move(&BB::POSSIBLE_MOVES[*i]))
                                .map(|&x| x)
                                .collect();
                            let rand_num = rng.gen_range(0..indices.len());
                            gb.make_move(BB::POSSIBLE_MOVES[indices[rand_num]]);
                            if x_turn {
                                x_moves.push(rand_num);
                            } else {
                                o_moves.push(rand_num);
                            }
                        }
                        x_turn = !x_turn
                    }

                    // game ended
                    match gb.game_state() {
                        GS::XWin => {
                            if !is_playing_self {
                                //playing random
                                if train_x {
                                    network_wins += 1;
                                    let vec: Vec<(Vec<bool>, Vec<f32>)> =
                                        x_states.iter().cloned().zip(vec_dCda_x).collect();
                                    network.train(vec, 2);
                                    network.update();
                                    x_trained += 1;
                                } else {
                                    random_wins += 1;
                                    //network lost
                                    #[allow(non_snake_case)]
                                    for dCda_o in vec_dCda_o.iter_mut() {
                                        *dCda_o = dCda_o.iter().map(|x| *x * -1.0).collect();
                                    }

                                    let vec: Vec<(Vec<bool>, Vec<f32>)> =
                                        o_states.iter().cloned().zip(vec_dCda_o).collect();
                                    network.train(vec, 2);
                                    network.update();
                                    o_trained += 1;
                                }
                            } else {
                                //playing self
                                let vec: Vec<(Vec<bool>, Vec<f32>)> =
                                    x_states.iter().cloned().zip(vec_dCda_x).collect();
                                network.train(vec, 2);

                                #[allow(non_snake_case)]
                                for dCda_o in vec_dCda_o.iter_mut() {
                                    *dCda_o = dCda_o.iter().map(|x| *x * -1.0).collect();
                                }

                                let vec: Vec<(Vec<bool>, Vec<f32>)> =
                                    o_states.iter().cloned().zip(vec_dCda_o).collect();
                                network.train(vec, 2);
                                network.update();
                                self_play += 1;
                            }
                        }
                        GS::OWin => {
                            if !is_playing_self {
                                if train_x {
                                    random_wins += 1;
                                    //network lost
                                    #[allow(non_snake_case)]
                                    for dCda_x in vec_dCda_x.iter_mut() {
                                        *dCda_x = dCda_x.iter().map(|x| *x * -1.0).collect();
                                    }

                                    let vec: Vec<(Vec<bool>, Vec<f32>)> =
                                        x_states.iter().cloned().zip(vec_dCda_x).collect();
                                    network.train(vec, 2);
                                    network.update();
                                    x_trained += 1;
                                } else {
                                    network_wins += 1;
                                    let vec: Vec<(Vec<bool>, Vec<f32>)> =
                                        o_states.iter().cloned().zip(vec_dCda_o).collect();
                                    network.train(vec, 2);
                                    network.update();
                                    o_trained += 1;
                                }
                            } else {
                                //playing self
                                #[allow(non_snake_case)]
                                for dCda_x in vec_dCda_x.iter_mut() {
                                    *dCda_x = dCda_x.iter().map(|x| *x * -1.0).collect();
                                }

                                let vec: Vec<(Vec<bool>, Vec<f32>)> =
                                    x_states.iter().cloned().zip(vec_dCda_x).collect();
                                network.train(vec, 2);

                                let vec: Vec<(Vec<bool>, Vec<f32>)> =
                                    o_states.iter().cloned().zip(vec_dCda_o).collect();
                                network.train(vec, 2);
                                network.update();
                                self_play += 1
                            }
                        }
                        GS::Tie => {
                            if !is_playing_self {
                                draws += 1
                            } else {
                                // here to train draws against self.
                            }
                        }
                        _ => panic!("game state should be impossible!"),
                    }
                    if now.elapsed() >= Duration::from_secs(5) {
                        //update every 15 sec
                        now = Instant::now();
                        let total: u128 = network_wins + random_wins + draws;
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
                            (network_wins as f64) / (total as f64),
                            (random_wins as f64) / (total as f64),
                            (draws as f64) / (total as f64)
                        )?;
                        writeln!(
                            stream,
                            "Time Training: {:.2?}, Trained: {}k, (X:{},O:{}), Self Play: {}",
                            start_time.elapsed(),
                            (x_trained + o_trained) / 1000,
                            x_trained,
                            o_trained,
                            self_play
                        )?;
                        stream.flush()?;
                    }
                    train_x = rng.gen();
                    is_playing_self = rng.gen();
                }
            }
            CS::Sample => unimplemented!("oops"),
            CS::Play => unimplemented!("oops"),
            _ => panic!("choice state impossible"),
        }
    }
    Ok(())
}
