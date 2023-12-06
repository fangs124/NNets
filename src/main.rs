#![allow(non_camel_case_types, non_snake_case, dead_code)]
mod gameboards;
mod nnets;

use crate::gameboards::*;

use crate::nnets::*;
use itertools::izip;

use crossterm::{cursor, ExecutableCommand};
use crossterm::terminal::{Clear,ClearType};
use inquire::Select;
use rand::Rng;
use std::env;
use std::fmt;
use std::fmt::Display;
use std::fs::File;
#[allow(unused_imports)]
use std::io::{stderr, stdout, BufReader, BufWriter, Read, Write};
use std::time:: Instant;
use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;

#[derive(PartialEq, Copy, Clone)]
struct ScoreBoard {
    invalid_count: u64,
    vs_random: u64,
    vs_self: u64,
    net_wins: u64,
    random_wins: u64,
    draws: u64,
    self_plays: u64,
    self_draws: u64,
    prev_w: f64,
    prev_l: f64,
    prev_d: f64,
    epoch: u32,
    start_time: Instant,
    now: Instant,
}
impl ScoreBoard {
    fn new() -> Self {
        ScoreBoard {
            invalid_count: 0,
            vs_random: 0,
            vs_self: 0,
            net_wins: 0,
            random_wins: 0,
            draws: 0,
            self_plays: 0,
            self_draws: 0,
            prev_w: 0.0,
            prev_l: 0.0,
            prev_d: 0.0,
            epoch: 0,
            start_time: Instant::now(),
            now: Instant::now(),
        }

    }
    fn update(&mut self) {
        self.prev_w = 100.0 * (self.net_wins as f64) / (self.vs_random as f64);
        self.prev_l = 100.0 * (self.random_wins as f64) / (self.vs_random as f64);
        self.prev_d = 100.0 * (self.draws as f64) / (self.vs_random as f64);
        self.self_plays = 0;
        self.net_wins = 0;
        self.random_wins = 0;
        self.draws = 0;
        self.self_draws = 0;
        self.invalid_count = 0;
        self.now = Instant::now();
    }
    fn write_to_buf<T: Write>(&mut self, stream: &mut BufWriter<T>) -> std::io::Result<()> {
        writeln!(
            stream,
            "N wins: {}, R wins: {}, Draws: {}, [{:.2}+({:.2}):{:.2}+({:.2}):{:.2}+({:.2})] (100k-epoch: {}, {:.2?}) invalid this epoch: {}",
            self.net_wins,
            self.random_wins,
            self.draws,
            100.0 * (self.net_wins as f64) / (self.vs_random as f64),
            (100.0 * (self.net_wins as f64) / (self.vs_random as f64)) - self.prev_w as f64,
            100.0 * (self.random_wins as f64) / (self.vs_random as f64),
            (100.0 * (self.random_wins as f64) / (self.vs_random as f64)) - self.prev_l as f64, 
            100.0 * (self.draws as f64) / (self.vs_random as f64),
            (100.0 * (self.draws as f64) / (self.vs_random as f64)) - self.prev_d as f64,
            self.epoch,
            self.now.elapsed(),
            self.invalid_count,
        )?;
        writeln!(
            stream,
            "Time Training: {:.2?}, Self Play: {} (Draws: {:.2}%)",
            self.start_time.elapsed(),
            self.self_plays,
            (self.self_draws as f64 / self.self_plays as f64) * 100.0,
        )?;
        stream.flush()?;
        Ok(())
    }
}

type SB = ScoreBoard;

impl InputType for bool {
    fn to_f64(&self) -> f64 {
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
    pub fn to_vec_bool_x(&self) -> Vec<bool> {
        let mut vec: Vec<bool> = Vec::new();
        vec.append(&mut self.x_b.to_vec_bool());
        vec.append(&mut self.o_b.to_vec_bool());
        vec
    }
    pub fn to_vec_bool_o(&self) -> Vec<bool> {
        let mut vec: Vec<bool> = Vec::new();
        vec.append(&mut self.o_b.to_vec_bool());
        vec.append(&mut self.x_b.to_vec_bool());
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

// some constants/parameters
const BATCH_SIZE: usize = 1000;
fn main() -> std::io::Result<()> {
    env::set_var("RUST_BACKTRACE", "1");

    let mut choice: CS = CS::NoChoice;
    let start_q = String::from("NNRL v0.1:");
    let start_opt: Vec<CS> = vec![
        CS::Play,
        CS::Train,
        CS::Sample,
        CS::Quit,
        /* Sample Game,  Save NNet, Load NNet, Print NNet -- to add later?*/
    ];
    let mut is_quit = false;

    /* ==== network loading/generating ==== */
    let network_q = String::from("New Network?");
    let network_opt = vec!["New Network", "Load Network"];
    let node_count: Vec<usize> = vec![18, 36, 18, 9]; //param_here
    let mut gb = GameBoard::NEW_GAMEBOARD;
    let mut network: Network<bool>;

    match Select::new(&network_q, network_opt)
        .prompt()
        .expect("Prompt error")
    {
        "New Network" => {
            let now = Instant::now();
            println!("Initializing...");
            network = Network::new_default(node_count.clone(), gb.to_vec_bool_o());
            println!("Elapsed: {:.2?}", now.elapsed());
        }
        "Load Network" => {
            let now = Instant::now();
            println!("Loading...");
            let file = File::open(format!("{:?}network.json", node_count.clone()))?;
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

    /* ==== main loop ==== */
    while !is_quit {
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
                        let file = File::create(format!("{:?}network.json", node_count.clone()))?;

                        serde_json::to_writer(file, &network)?;
                    }
                    "No" => {}
                    _ => panic!("Prompt error"),
                }
                is_quit = true;
                continue;
            }
            CS::Train => {
                let mut rng = rand::thread_rng();
                let stdout = stdout();
                let mut stream_out: BufWriter<&std::io::Stdout> = BufWriter::new(&stdout);
                //let mut stream_err: BufWriter<&std::io::Stderr> = BufWriter::new(&stderr());
                writeln!(stream_out, "Press q to stop.\n\n")?;

                let mut sb = SB::new();

                /* training flow controls */
                let mut is_training = true;
                let mut is_playing_self = false;
                let mut loop_counter: usize = 0;
                let f = File::create(format!("{:?}log.txt", node_count.clone()))
                    .expect("file creation failed");
                let mut f_buff: BufWriter<&File> = BufWriter::new(&f);
                sb.write_to_buf(&mut stream_out)?;

                while is_training {
                    // listen to 'q' for interupt
                    let return_val = unsafe { GetAsyncKeyState(0x51 as i32) };
                    if return_val & 0x01 != 0 {
                        //stop training
                        choice = CS::NoChoice;
                        is_training = false;
                        continue;
                    };
                    if is_playing_self {
                        net_vs_self(&mut network, &mut gb, &mut sb, true)
                    } else {
                        net_vs_random(&mut network, &mut gb, &mut sb, true)
                    }

                    is_playing_self = rng.gen();
                    loop_counter += 1;

                    if (loop_counter % BATCH_SIZE) == 0 {
                        network.update();
                    }
                    if (loop_counter % BATCH_SIZE*100) == 0 {
                        //100K
                        sb.epoch += 1;
                        stream_out.execute(cursor::MoveUp(2)).expect("xcross error");
                        stream_out
                            .execute(Clear(ClearType::FromCursorDown))
                            .expect("xcross error");
                        stream_out.flush()?;
                        sb.write_to_buf(&mut stream_out)?;
                        sb.write_to_buf(&mut f_buff)?;
                        f_buff.flush()?;
                        
                        sb.update();
                    }  
                }
            }
            CS::Sample => unimplemented!("oops"),
            CS::Play => unimplemented!("oops"),
            _ => panic!("choice state impossible"),
        }
    }
    Ok(())
}

fn net_vs_self(net: &mut Network<bool>, gb: &mut GB, sb: &mut SB, is_train: bool) {
    gb.new_game();

    let mut x_turn: bool = true; // x goes first
    let mut x_moves: Vec<usize> = Vec::new();
    let mut o_moves: Vec<usize> = Vec::new();
    let mut x_states: Vec<Vec<bool>> = Vec::new();
    let mut o_states: Vec<Vec<bool>> = Vec::new();

    #[allow(non_snake_case)]
    let mut vec_dCda_x: Vec<Vec<f64>> = Vec::new();
    #[allow(non_snake_case)]
    let mut vec_dCda_o: Vec<Vec<f64>> = Vec::new();

    while gb.game_state() == GS::Ongoing {
        #[allow(non_snake_case)]
        let mut dCda: Vec<f64> = vec![0.0; BB::MOVES.len()];
        let mut vec_bool: Vec<bool> = Vec::new();

        if x_turn {
            vec_bool.append(&mut gb.to_vec_bool_x());
            x_states.push(vec_bool.clone());
        } else {
            vec_bool.append(&mut gb.to_vec_bool_o());
            o_states.push(vec_bool.clone());
        }

        let mut inv_states: Vec<Vec<bool>> = Vec::new();
        let mut inv_moves: Vec<usize> = Vec::new();
        let output: Vec<f64> = net.forward_prop(&mut vec_bool);

        // teach network not to make invalid moves
        let inv_indices = get_invalid_indices(&output, 0.01, &gb);
        let count = inv_indices.len();

        for i in &inv_indices {
            dCda[*i] = (-0.75) * inv_indices.len() as f64;
            inv_states.push(vec_bool.clone());
            inv_moves.push(*i);
        }

        sb.invalid_count += inv_indices.len() as u64;

        // zip to pass to train
        if is_train {
            let vec = izip!(inv_states, vec![dCda; count], inv_moves);
            net.train(vec.collect(), 2)
        };

        let index = get_index(&output, gb);
        gb.make_move(BB::MOVES[index])
            .expect("net_vs_self: invalid move");

        // reset dCda for x_move
        dCda = vec![0.0; BB::MOVES.len()];
        dCda[index] = 1.0;

        if x_turn {
            x_moves.push(index);
            vec_dCda_x.push(dCda);
        } else {
            o_moves.push(index);
            vec_dCda_o.push(dCda);
        }
        // pass turn to next player
        x_turn = !x_turn;
    }

    // game ended
    if is_train {
        match gb.game_state() {
            GS::XWin => {
                for dCda_o in vec_dCda_o.iter_mut() {
                    *dCda_o = dCda_o.iter().map(|x| *x * -1.0).collect();
                }
            }
            GS::OWin => {
                for dCda_x in vec_dCda_x.iter_mut() {
                    *dCda_x = dCda_x.iter().map(|x| *x * -1.0).collect();
                }
            }
            GS::Tie => return,
            _ => panic!("net_vs_random: state is impossible"),
        }

        let vecx = izip!(x_states, vec_dCda_x, x_moves).collect();
        let veco = izip!(o_states, vec_dCda_o, o_moves).collect();
        
        net.train(vecx, 2);
        net.train(veco, 2);
    }
    sb.self_plays += 1;
}

type N = Network<bool>;
fn net_vs_net(net1: &mut N, net2: &mut N, gb: &mut GB, sb: &mut SB, is_train: bool) {
    gb.new_game();
    let mut rng = rand::thread_rng();
    let net1_is_x: bool = rng.gen();

    let mut x_turn: bool = true; // x goes first
    let mut x_moves: Vec<usize> = Vec::new();
    let mut o_moves: Vec<usize> = Vec::new();
    let mut x_states: Vec<Vec<bool>> = Vec::new();
    let mut o_states: Vec<Vec<bool>> = Vec::new();

    #[allow(non_snake_case)]
    let mut vec_dCda_x: Vec<Vec<f64>> = Vec::new();
    #[allow(non_snake_case)]
    let mut vec_dCda_o: Vec<Vec<f64>> = Vec::new();

    while gb.game_state() == GS::Ongoing {
        #[allow(non_snake_case)]
        let mut dCda: Vec<f64> = vec![0.0; BB::MOVES.len()];
        let mut vec_bool: Vec<bool> = Vec::new();

        if x_turn {
            vec_bool.append(&mut gb.to_vec_bool_x());
            x_states.push(vec_bool.clone());
        } else {
            vec_bool.append(&mut gb.to_vec_bool_o());
            o_states.push(vec_bool.clone());
        }

        // net1's turn
        if x_turn == net1_is_x {
            let mut inv_states: Vec<Vec<bool>> = Vec::new();
            let mut inv_moves: Vec<usize> = Vec::new();
            let output: Vec<f64> = net1.forward_prop(&mut vec_bool);

            // teach network not to make invalid moves
            let inv_indices = get_invalid_indices(&output, 0.01, &gb);
            let count = inv_indices.len();

            for i in &inv_indices {
                dCda[*i] = (-0.75) * inv_indices.len() as f64;
                inv_states.push(vec_bool.clone());
                inv_moves.push(*i);
            }

            sb.invalid_count += inv_indices.len() as u64;

            // zip to pass to train
            if is_train {
                let vec = izip!(inv_states, vec![dCda; count], inv_moves);
                net1.train(vec.collect(), 2)
            };

            let index = get_index(&output, gb);
            gb.make_move(BB::MOVES[index])
                .expect("net_vs_net: invalid move");

            // reset dCda for x_move
            dCda = vec![0.0; BB::MOVES.len()];
            dCda[index] = 1.0;

            if x_turn {
                x_moves.push(index);
                vec_dCda_x.push(dCda);
            } else {
                o_moves.push(index);
                vec_dCda_o.push(dCda);
            }
        }
        // net2's turn
        else {
            let mut inv_states: Vec<Vec<bool>> = Vec::new();
            let mut inv_moves: Vec<usize> = Vec::new();
            let output: Vec<f64> = net2.forward_prop(&mut vec_bool);

            // teach network not to make invalid moves
            let inv_indices = get_invalid_indices(&output, 0.01, &gb);
            let count = inv_indices.len();

            for &i in &inv_indices {
                dCda[i] = (-0.75) * inv_indices.len() as f64;
                inv_states.push(vec_bool.clone());
                inv_moves.push(i);
            }

            sb.invalid_count += inv_indices.len() as u64;

            // zip to pass to train
            if is_train {
                let vec = izip!(inv_states, vec![dCda; count], inv_moves);
                net2.train(vec.collect(), 2)
            };

            let index = get_index(&output, gb);
            gb.make_move(BB::MOVES[index])
                .expect("net_vs_random: invalid move");

            // reset dCda for x_move
            dCda = vec![0.0; BB::MOVES.len()];
            dCda[index] = 1.0;

            if x_turn {
                x_moves.push(index);
                vec_dCda_x.push(dCda);
            } else {
                o_moves.push(index);
                vec_dCda_o.push(dCda);
            }
        }

        // pass turn to next player
        x_turn = !x_turn;
    }

    // game ended
    if is_train {
        match gb.game_state() {
            GS::XWin => {
                for dCda_o in vec_dCda_o.iter_mut() {
                    *dCda_o = dCda_o.iter().map(|x| *x * -1.0).collect();
                }
            }
            GS::OWin => {
                for dCda_x in vec_dCda_x.iter_mut() {
                    *dCda_x = dCda_x.iter().map(|x| *x * -1.0).collect();
                }
            }
            GS::Tie => return,
            _ => panic!("net_vs_random: state is impossible"),
        }

        let vecx = izip!(x_states, vec_dCda_x, x_moves).collect();
        let veco = izip!(o_states, vec_dCda_o, o_moves).collect();
        if net1_is_x {
            net1.train(vecx, 2);
            net2.train(veco, 2);
        } else {
            net1.train(veco, 2);
            net2.train(vecx, 2);
        }
    }
}

fn net_vs_random(net: &mut Network<bool>, gb: &mut GameBoard, sb: &mut SB, is_train: bool) {
    gb.new_game();
    let mut rng = rand::thread_rng();
    let net_is_x: bool = rng.gen();

    let mut x_turn: bool = true; // x goes first
    let mut x_moves: Vec<usize> = Vec::new();
    let mut o_moves: Vec<usize> = Vec::new();
    let mut x_states: Vec<Vec<bool>> = Vec::new();
    let mut o_states: Vec<Vec<bool>> = Vec::new();

    #[allow(non_snake_case)]
    let mut vec_dCda_x: Vec<Vec<f64>> = Vec::new();
    #[allow(non_snake_case)]
    let mut vec_dCda_o: Vec<Vec<f64>> = Vec::new();

    while gb.game_state() == GS::Ongoing {
        #[allow(non_snake_case)]
        let mut dCda: Vec<f64> = vec![0.0; BB::MOVES.len()];
        let mut vec_bool: Vec<bool> = Vec::new();

        if x_turn {
            vec_bool.append(&mut gb.to_vec_bool_x());
            x_states.push(vec_bool.clone());
        } else {
            vec_bool.append(&mut gb.to_vec_bool_o());
            o_states.push(vec_bool.clone());
        }

        // net's turn
        if x_turn == net_is_x {
            let mut inv_states: Vec<Vec<bool>> = Vec::new();
            let mut inv_moves: Vec<usize> = Vec::new();
            let output: Vec<f64> = net.forward_prop(&mut vec_bool);

            // teach network not to make invalid moves
            let inv_indices = get_invalid_indices(&output, 0.01, &gb);
            let count = inv_indices.len();

            for i in &inv_indices {
                dCda[*i] = (-0.75) * inv_indices.len() as f64;
                inv_states.push(vec_bool.clone());
                inv_moves.push(*i);
            }

            sb.invalid_count += inv_indices.len() as u64;

            // zip to pass to train
            if is_train {
                let vec = izip!(inv_states, vec![dCda; count], inv_moves);
                net.train(vec.collect(), 2)
            };

            let index = get_index(&output, gb);
            gb.make_move(BB::MOVES[index])
                .expect("net_vs_random: invalid move");

            // reset dCda for x_move
            dCda = vec![0.0; BB::MOVES.len()];
            dCda[index] = 1.0;

            if x_turn {
                x_moves.push(index);
                vec_dCda_x.push(dCda);
            } else {
                o_moves.push(index);
                vec_dCda_o.push(dCda);
            }
        }
        // random's turn
        else {
            // make a random valid move
            let indices: Vec<usize> = (0..9)
                .filter(|&i| gb.is_valid_move(&BB::MOVES[i]))
                .collect();
            let rand_num = rng.gen_range(0..indices.len());
            gb.make_move(BB::MOVES[indices[rand_num]])
                .expect("net_vs_random: invalid move");

            dCda[indices[rand_num]] = 1.0;
            if x_turn {
                x_moves.push(indices[rand_num]);
                vec_dCda_x.push(dCda);
            } else {
                o_moves.push(indices[rand_num]);
                vec_dCda_o.push(dCda);
            }
        }

        // pass turn to next player
        x_turn = !x_turn;
    }

    if gb.game_state() == GS::Tie {
        sb.draws += 1
    } else if (gb.game_state() == GS::XWin) == net_is_x {
        sb.net_wins += 1
    } else {
        sb.random_wins += 1
    }
    // game ended
    if is_train {
        match gb.game_state() {
            GS::XWin => {
                for dCda_o in vec_dCda_o.iter_mut() {
                    *dCda_o = dCda_o.iter().map(|x| *x * -1.0).collect();
                }
            }
            GS::OWin => {
                for dCda_x in vec_dCda_x.iter_mut() {
                    *dCda_x = dCda_x.iter().map(|x| *x * -1.0).collect();
                }
            }
            GS::Tie => return,
            _ => panic!("net_vs_random: state is impossible"),
        }

        let vec = izip!(x_states, vec_dCda_x, x_moves).collect();
        net.train(vec, 2);
        let vec = izip!(o_states, vec_dCda_o, o_moves).collect();
        net.train(vec, 2);
    }
}

// returns the indices of all invalid moves by the network that's above the treshold
fn get_invalid_indices(net_output: &Vec<f64>, treshold: f64, gb: &GameBoard) -> Vec<usize> {
    net_output
        .iter()
        .enumerate()
        .filter(|&(_, &x)| x >= treshold)
        .filter(|&(i, _)| !gb.is_valid_move(&BB::MOVES[i]))
        .map(|(index, _)| index)
        .collect()
}

// returns the index of a valid move corresponding to the network's highest output
fn get_index(net_output: &Vec<f64>, gb: &GameBoard) -> usize {
    net_output
        .iter()
        .enumerate()
        .filter(|&(i, _)| gb.is_valid_move(&BB::MOVES[i]))
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .expect("get_index fail")
}

