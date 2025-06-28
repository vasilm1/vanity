use clap::Parser;
use logfather::{Level, Logger};
use num_format::{Locale, ToFormattedString};
use rand::{distributions::Alphanumeric, Rng, RngCore};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use sha2::{Digest, Sha256};
use solana_pubkey::Pubkey;
use ed25519_dalek::{SigningKey, VerifyingKey};
#[cfg(feature = "deploy")]
use {
    solana_rpc_client::rpc_client::RpcClient,
    solana_sdk::{
        bpf_loader_upgradeable::{self, get_program_data_address, UpgradeableLoaderState},
        instruction::{AccountMeta, Instruction},
        loader_upgradeable_instruction::UpgradeableLoaderInstruction,
        signature::read_keypair_file,
        signer::Signer,
        system_instruction, system_program, sysvar,
        transaction::Transaction,
    },
    std::path::PathBuf,
};

use std::{
    array,
    str::FromStr,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

#[derive(Debug, Parser)]
pub enum Command {
    /// Generate a vanity address using CreateAccountWithSeed (deterministic)
    Grind(GrindArgs),
    /// Generate a random keypair with a vanity public key
    Keypair(KeypairArgs),
    /// Verify that a seed generates the expected address
    Verify(VerifyArgs),
    #[cfg(feature = "deploy")]
    /// Deploy a program using a vanity address
    Deploy(DeployArgs),
}

#[derive(Debug, Parser)]
pub struct GrindArgs {
    /// The pubkey that will be the signer for the CreateAccountWithSeed instruction
    #[clap(long, value_parser = parse_pubkey)]
    pub base: Pubkey,

    /// The account owner, e.g. BPFLoaderUpgradeab1e11111111111111111111111 or TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA
    #[clap(long, value_parser = parse_pubkey)]
    pub owner: Pubkey,

    /// The target prefix for the pubkey
    #[clap(long)]
    pub prefix: Option<String>,

    #[clap(long)]
    pub suffix: Option<String>,

    /// Whether user cares about the case of the pubkey
    #[clap(long, default_value_t = false)]
    pub case_insensitive: bool,

    /// Optional log file
    #[clap(long)]
    pub logfile: Option<String>,

    /// Number of gpus to use for mining
    #[clap(long, default_value_t = 1)]
    #[cfg(feature = "gpu")]
    pub num_gpus: u32,

    /// Number of cpu threads to use for mining
    #[clap(long, default_value_t = 0)]
    pub num_cpus: u32,
}

#[derive(Debug, Parser)]
#[command(about = "Generate random keypairs with vanity public keys")]
#[command(long_about = "Generate completely random Ed25519 keypairs where the public key matches your desired prefix/suffix. This produces a real private key you can use directly, unlike the seed-based approach which requires CreateAccountWithSeed.")]
pub struct KeypairArgs {
    /// The target prefix for the public key (case-sensitive by default)
    #[clap(long)]
    #[clap(help = "Desired prefix for the vanity public key (e.g., 'ABC' to get keys starting with ABC)")]
    pub prefix: Option<String>,

    /// The target suffix for the public key (case-sensitive by default)
    #[clap(long)]
    #[clap(help = "Desired suffix for the vanity public key (e.g., 'xyz' to get keys ending with xyz)")]
    pub suffix: Option<String>,

    /// Ignore case when matching prefix/suffix
    #[clap(long, default_value_t = false)]
    #[clap(help = "Make prefix/suffix matching case-insensitive")]
    pub case_insensitive: bool,

    /// Path to log mining progress and results
    #[clap(long)]
    #[clap(help = "Optional file to log mining progress and found keypairs")]
    pub logfile: Option<String>,

    /// Number of GPUs to use for mining (requires GPU feature)
    #[clap(long, default_value_t = 8)]
    #[cfg(feature = "gpu")]
    #[clap(help = "Number of GPUs to use for parallel keypair mining (default: 8)")]
    pub num_gpus: u32,

    /// Number of CPU threads to use for mining (0 = auto-detect)
    #[clap(long, default_value_t = 0)]
    #[clap(help = "Number of CPU threads for mining (0 = auto-detect based on CPU cores, set to 0 to disable CPU mining when using GPUs)")]
    pub num_cpus: u32,
}

#[derive(Debug, Parser)]
pub struct VerifyArgs {
    /// The pubkey that will be the signer for the CreateAccountWithSeed instruction
    #[clap(long, value_parser = parse_pubkey)]
    pub base: Pubkey,

    /// The account owner, e.g. BPFLoaderUpgradeab1e11111111111111111111111 or TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA
    #[clap(long, value_parser = parse_pubkey)]
    pub owner: Pubkey,

    /// The seed to verify
    #[clap(long)]
    pub seed: String,
}

#[cfg(feature = "deploy")]
#[derive(Debug, Parser)]
pub struct DeployArgs {
    /// The keypair that will be the signer for the CreateAccountWithSeed instruction
    #[clap(long)]
    pub base: PathBuf,

    /// The keypair that will be the signer for the CreateAccountWithSeed instruction
    #[clap(long, default_value = "https://api.mainnet-beta.solana.com")]
    pub rpc: String,

    /// The account owner, e.g. BPFLoaderUpgradeab1e11111111111111111111111 or TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA
    #[clap(long, value_parser = parse_pubkey)]
    pub owner: Pubkey,

    /// Buffer where the program has been written (via solana program write-buffer)
    #[clap(long, value_parser = parse_pubkey)]
    pub buffer: Pubkey,

    /// Path to keypair that will pay for deploy. when this is None, base is used as payer
    #[clap(long)]
    pub payer: Option<PathBuf>,

    /// Seed grinded via grind
    #[clap(long)]
    pub seed: String,

    /// Program authority (default is (payer) keypair's pubkey)
    #[clap(long)]
    pub authority: Option<Pubkey>,

    /// Compute unit price
    #[clap(long)]
    pub compute_unit_price: Option<u64>,

    /// Optional log file
    #[clap(long)]
    pub logfile: Option<String>,
}

static EXIT: AtomicBool = AtomicBool::new(false);

fn main() {
    rayon::ThreadPoolBuilder::new().build_global().unwrap();

    // Parse command line arguments
    let command = Command::parse();
    match command {
        Command::Grind(args) => {
            grind(args);
        }

        Command::Keypair(args) => {
            grind_keypair(args);
        }

        Command::Verify(args) => {
            verify(args);
        }

        #[cfg(feature = "deploy")]
        Command::Deploy(args) => {
            deploy(args);
        }
    }
}

fn verify(args: VerifyArgs) {
    // Unpack create with seed arguments
    let VerifyArgs { base, owner, seed } = args;

    let result = Pubkey::create_with_seed(&base, &seed, &owner).unwrap();
    println!("Results:");
    println!("  base  {base}");
    println!("  owner {owner}");
    println!("  seed  {seed}\n");
    println!("  resulting pubkey: {result}")
}

#[cfg(feature = "deploy")]
fn deploy(args: DeployArgs) {
    // Load base and payer keypair
    let base_keypair = read_keypair_file(&args.base).expect("failed to read base keypair");
    let payer_keypair = args
        .payer
        .as_ref()
        .map(|payer| read_keypair_file(payer).expect("failed to read payer keypair"))
        .unwrap_or(base_keypair.insecure_clone());
    let authority = args.authority.unwrap_or_else(|| payer_keypair.pubkey());

    // Target
    let target = Pubkey::create_with_seed(&base_keypair.pubkey(), &args.seed, &args.owner).unwrap();
    // Fetch rent
    let rpc_client = RpcClient::new(args.rpc);
    // this is such a dumb way to do this
    let buffer_len = rpc_client.get_account_data(&args.buffer).unwrap().len();
    // I forgot the header len so let's just add 64 for now lol
    let rent = rpc_client
        .get_minimum_balance_for_rent_exemption(UpgradeableLoaderState::size_of_program())
        .expect("failed to fetch rent");

    // Create account with seed
    let instructions = deploy_with_max_program_len_with_seed(
        &payer_keypair.pubkey(),
        &target,
        &args.buffer,
        &authority,
        rent,
        64 + buffer_len,
        &base_keypair.pubkey(),
        &args.seed,
    );
    // Transaction
    let blockhash = rpc_client.get_latest_blockhash().unwrap();
    let signers = if args.payer.is_none() {
        vec![&base_keypair]
    } else {
        vec![&base_keypair, &payer_keypair]
    };
    let transaction = Transaction::new_signed_with_payer(
        &instructions,
        Some(&payer_keypair.pubkey()),
        &signers,
        blockhash,
    );

    let sig = rpc_client
        .send_and_confirm_transaction(&transaction)
        .unwrap();
    println!("Deployed {target}: {sig}");
}

#[cfg(feature = "deploy")]
pub fn deploy_with_max_program_len_with_seed(
    payer_address: &Pubkey,
    program_address: &Pubkey,
    buffer_address: &Pubkey,
    upgrade_authority_address: &Pubkey,
    program_lamports: u64,
    max_data_len: usize,
    base: &Pubkey,
    seed: &str,
) -> [Instruction; 2] {
    let programdata_address = get_program_data_address(program_address);
    [
        system_instruction::create_account_with_seed(
            payer_address,
            program_address,
            base,
            seed,
            program_lamports,
            UpgradeableLoaderState::size_of_program() as u64,
            &bpf_loader_upgradeable::id(),
        ),
        Instruction::new_with_bincode(
            bpf_loader_upgradeable::id(),
            &UpgradeableLoaderInstruction::DeployWithMaxDataLen { max_data_len },
            vec![
                AccountMeta::new(*payer_address, true),
                AccountMeta::new(programdata_address, false),
                AccountMeta::new(*program_address, false),
                AccountMeta::new(*buffer_address, false),
                AccountMeta::new_readonly(sysvar::rent::id(), false),
                AccountMeta::new_readonly(sysvar::clock::id(), false),
                AccountMeta::new_readonly(system_program::id(), false),
                AccountMeta::new_readonly(*upgrade_authority_address, true),
            ],
        ),
    ]
}

fn grind(mut args: GrindArgs) {
    maybe_update_num_cpus(&mut args.num_cpus);
    let prefix = get_validated_prefix(&args);
    let suffix = get_validated_suffix(&args);

    // Initialize logger with optional logfile
    let mut logger = Logger::new();
    if let Some(ref logfile) = args.logfile {
        logger.file(true);
        logger.path(logfile);
    }

    // Slightly more compact log format
    logger.log_format("[{timestamp} {level}] {message}");
    logger.timestamp_format("%Y-%m-%d %H:%M:%S");
    logger.level(Level::Info);

    // Print resource usage
    logfather::info!("using {} threads", args.num_cpus);
    #[cfg(feature = "gpu")]
    logfather::info!("using {} gpus", args.num_gpus);

    #[cfg(feature = "gpu")]
    let _gpu_threads: Vec<_> = (0..args.num_gpus)
        .map(move |gpu_index| {
            std::thread::Builder::new()
                .name(format!("gpu{gpu_index}"))
                .spawn(move || {
                    logfather::trace!("starting gpu {gpu_index}");

                    let mut out = [0; 24];
                    for iteration in 0_u64.. {
                        // Exit if a thread found a solution
                        if EXIT.load(Ordering::SeqCst) {
                            logfather::trace!("gpu thread {gpu_index} exiting");
                            return;
                        }

                        // Generate new seed for this gpu & iteration
                        let seed = new_gpu_seed(gpu_index, iteration);
                        let timer = Instant::now();
                        unsafe {
                            vanity_round(gpu_index, seed.as_ref().as_ptr(), args.base.to_bytes().as_ptr(), args.owner.to_bytes().as_ptr(), prefix.as_ptr(), suffix.as_ptr(), prefix.len() as u64, suffix.len() as u64,out.as_mut_ptr(), args.case_insensitive);
                        }
                        let time_sec = timer.elapsed().as_secs_f64();

                        // Reconstruct solution
                        let reconstructed: [u8; 32] = Sha256::new()
                            .chain_update(&args.base)
                            .chain_update(&out[..16])
                            .chain_update(&args.owner)
                            .finalize()
                            .into();
                        let out_str = fd_bs58::encode_32(reconstructed);
                        let out_str_target_check = maybe_bs58_aware_lowercase(&out_str, args.case_insensitive);
                        let count = u64::from_le_bytes(array::from_fn(|i| out[16 + i]));
                        logfather::info!(
                            "{} found in {:.3} seconds on gpu {gpu_index:>3}; {:>13} iters; {:>12} iters/sec",
                            &out_str,
                            time_sec,
                            count.to_formatted_string(&Locale::en),
                            ((count as f64 / time_sec) as u64).to_formatted_string(&Locale::en)
                        );

                        if out_str_target_check.starts_with(prefix) && out_str_target_check.ends_with(suffix) {
                            logfather::info!("out seed = {out:?} -> {}", core::str::from_utf8(&out[..16]).unwrap());
                            EXIT.store(true, Ordering::SeqCst);
                            logfather::trace!("gpu thread {gpu_index} exiting");
                            return;
                        }
                    }
                })
                .unwrap()
        })
        .collect();

    (0..args.num_cpus).into_par_iter().for_each(|i| {
        let timer = Instant::now();
        let mut count = 0_u64;

        let base_sha = Sha256::new().chain_update(args.base);
        loop {
            if EXIT.load(Ordering::Acquire) {
                return;
            }

            let mut seed_iter = rand::thread_rng().sample_iter(&Alphanumeric).take(16);
            let seed: [u8; 16] = array::from_fn(|_| seed_iter.next().unwrap());

            let pubkey_bytes: [u8; 32] = base_sha
                .clone()
                .chain_update(seed)
                .chain_update(args.owner)
                .finalize()
                .into();
            let pubkey = fd_bs58::encode_32(pubkey_bytes);
            let out_str_target_check = maybe_bs58_aware_lowercase(&pubkey, args.case_insensitive);

            count += 1;

            // Did cpu find target?
            if out_str_target_check.starts_with(prefix) && out_str_target_check.ends_with(suffix) {
                let time_secs = timer.elapsed().as_secs_f64();
                logfather::info!(
                    "cpu {i} found target: {pubkey}; {seed:?} -> {} in {:.3}s; {} attempts; {} attempts per second",
                    core::str::from_utf8(&seed).unwrap(),
                    time_secs,
                    count.to_formatted_string(&Locale::en),
                    ((count as f64 / time_secs) as u64).to_formatted_string(&Locale::en)
                );

                EXIT.store(true, Ordering::Release);
                break;
            }
        }
    });
}

fn get_validated_prefix(args: &GrindArgs) -> &'static str {
    // Static string of BS58 characters
    const BS58_CHARS: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    // Validate target (i.e. does it include 0, O, I, l)
    //
    // maybe TODO: technically we could accept I or o if case-insensitivity but I suspect
    // most users will provide lowercase targets for case-insensitive searches

    if let Some(ref prefix) = args.prefix {
        for c in prefix.chars() {
            assert!(
                BS58_CHARS.contains(c),
                "your prefix contains invalid bs58: {}",
                c
            );
        }
        let prefix = maybe_bs58_aware_lowercase(&prefix, args.case_insensitive);
        return prefix.leak();
    }
    ""
}

fn get_validated_suffix(args: &GrindArgs) -> &'static str {
    // Static string of BS58 characters
    const BS58_CHARS: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    // Validate target (i.e. does it include 0, O, I, l)
    //
    // maybe TODO: technically we could accept I or o if case-insensitivity but I suspect
    // most users will provide lowercase targets for case-insensitive searches

    if let Some(ref suffix) = args.suffix {
        for c in suffix.chars() {
            assert!(
                BS58_CHARS.contains(c),
                "your suffix contains invalid bs58: {}",
                c
            );
        }
        let suffix = maybe_bs58_aware_lowercase(&suffix, args.case_insensitive);
        return suffix.leak();
    }
    ""
}

fn maybe_bs58_aware_lowercase(target: &str, case_insensitive: bool) -> String {
    // L is only char that shouldn't be converted to lowercase in case-insensitivity case
    const LOWERCASE_EXCEPTIONS: &str = "L";

    if case_insensitive {
        target
            .chars()
            .map(|c| {
                if LOWERCASE_EXCEPTIONS.contains(c) {
                    c
                } else {
                    c.to_ascii_lowercase()
                }
            })
            .collect::<String>()
    } else {
        target.to_string()
    }
}

extern "C" {
    pub fn vanity_round(
        gpus: u32,
        seed: *const u8,
        base: *const u8,
        owner: *const u8,
        target: *const u8,
        suffix: *const u8,
        target_len: u64,
        suffix_len: u64,
        out: *mut u8,
        case_insensitive: bool,
    );
    
    #[cfg(feature = "gpu")]
    pub fn keypair_round(
        gpu_id: u32,
        seed: *const u8,
        target: *const u8,
        suffix: *const u8,
        target_len: u64,
        suffix_len: u64,
        out: *mut u8,
        case_insensitive: bool,
    );
}

#[cfg(feature = "gpu")]
fn new_gpu_seed(gpu_id: u32, iteration: u64) -> [u8; 32] {
    Sha256::new()
        .chain_update(rand::random::<[u8; 32]>())
        .chain_update(gpu_id.to_le_bytes())
        .chain_update(iteration.to_le_bytes())
        .finalize()
        .into()
}

fn parse_pubkey(input: &str) -> Result<Pubkey, String> {
    Pubkey::from_str(input).map_err(|e| e.to_string())
}

fn maybe_update_num_cpus(num_cpus: &mut u32) {
    if *num_cpus == 0 {
        *num_cpus = rayon::current_num_threads() as u32;
    }
}

fn grind_keypair(mut args: KeypairArgs) {
    maybe_update_num_cpus(&mut args.num_cpus);
    let prefix = get_validated_prefix_keypair(&args);
    let suffix = get_validated_suffix_keypair(&args);

    // Initialize logger with optional logfile
    let mut logger = Logger::new();
    if let Some(ref logfile) = args.logfile {
        logger.file(true);
        logger.path(logfile);
    }

    logger.log_format("[{timestamp} {level}] {message}");
    logger.timestamp_format("%Y-%m-%d %H:%M:%S");
    logger.level(Level::Info);

    logfather::info!("using {} threads for keypair grinding", args.num_cpus);
    #[cfg(feature = "gpu")]
    logfather::info!("using {} gpus for keypair grinding", args.num_gpus);
    logfather::info!("searching for prefix: '{}', suffix: '{}'", 
        args.prefix.as_deref().unwrap_or(""), 
        args.suffix.as_deref().unwrap_or(""));

    // GPU threads for keypair grinding
    #[cfg(feature = "gpu")]
    let _gpu_threads: Vec<_> = (0..args.num_gpus)
        .map(move |gpu_index| {
            std::thread::Builder::new()
                .name(format!("gpu{gpu_index}"))
                .spawn(move || {
                    logfather::trace!("starting gpu {gpu_index} for keypair grinding");

                    let mut out = [0; 40]; // 32 bytes private key + 8 bytes counter
                    for iteration in 0_u64.. {
                        // Exit if a thread found a solution
                        if EXIT.load(Ordering::SeqCst) {
                            logfather::trace!("gpu thread {gpu_index} exiting");
                            return;
                        }

                        // Generate new seed for this gpu & iteration
                        let seed = new_gpu_seed(gpu_index, iteration);
                        let timer = Instant::now();
                        unsafe {
                            keypair_round(gpu_index, seed.as_ref().as_ptr(), prefix.as_ptr(), suffix.as_ptr(), prefix.len() as u64, suffix.len() as u64, out.as_mut_ptr(), args.case_insensitive);
                        }
                        let time_sec = timer.elapsed().as_secs_f64();

                        // Extract private key and counter
                        let private_key_bytes: [u8; 32] = array::from_fn(|i| out[i]);
                        let count = u64::from_le_bytes(array::from_fn(|i| out[32 + i]));
                        
                        // Generate public key to verify and display
                        let signing_key = SigningKey::from_bytes(&private_key_bytes);
                        let verifying_key = signing_key.verifying_key();
                        let pubkey_bytes = verifying_key.to_bytes();
                        let pubkey = fd_bs58::encode_32(pubkey_bytes);
                        let pubkey_check = maybe_bs58_aware_lowercase(&pubkey, args.case_insensitive);
                        
                        logfather::info!(
                            "{} found in {:.3} seconds on gpu {gpu_index:>3}; {:>13} iters; {:>12} iters/sec",
                            &pubkey,
                            time_sec,
                            count.to_formatted_string(&Locale::en),
                            ((count as f64 / time_sec) as u64).to_formatted_string(&Locale::en)
                        );

                        if pubkey_check.starts_with(prefix) && pubkey_check.ends_with(suffix) {
                            let private_key_b58 = fd_bs58::encode_32(private_key_bytes);
                            logfather::info!("🎉 FOUND KEYPAIR! pubkey: {} private_key: {}", pubkey, private_key_b58);
                            EXIT.store(true, Ordering::SeqCst);
                            logfather::trace!("gpu thread {gpu_index} exiting");
                            return;
                        }
                    }
                })
                .unwrap()
        })
        .collect();

    // CPU threads for keypair grinding (as backup or when GPUs disabled)
    if args.num_cpus > 0 {
        (0..args.num_cpus).into_par_iter().for_each(|i| {
            let timer = Instant::now();
            let mut count = 0_u64;

            loop {
                if EXIT.load(Ordering::Acquire) {
                    return;
                }

                // Generate random 32-byte private key
                let mut private_key_bytes = [0u8; 32];
                rand::thread_rng().fill_bytes(&mut private_key_bytes);
                
                // Create signing key from random bytes
                let signing_key = SigningKey::from_bytes(&private_key_bytes);
                let verifying_key = signing_key.verifying_key();
                
                // Convert public key to Solana pubkey format
                let pubkey_bytes = verifying_key.to_bytes();
                let pubkey = fd_bs58::encode_32(pubkey_bytes);
                let pubkey_check = maybe_bs58_aware_lowercase(&pubkey, args.case_insensitive);

                count += 1;

                // Check if pubkey matches target
                if pubkey_check.starts_with(prefix) && pubkey_check.ends_with(suffix) {
                    let time_secs = timer.elapsed().as_secs_f64();
                    let private_key_b58 = fd_bs58::encode_32(private_key_bytes);
                    
                    logfather::info!(
                        "🎉 cpu {} found target keypair! pubkey: {} private_key: {} in {:.3}s; {} attempts; {} attempts per second",
                        i,
                        pubkey,
                        private_key_b58,
                        time_secs,
                        count.to_formatted_string(&Locale::en),
                        ((count as f64 / time_secs) as u64).to_formatted_string(&Locale::en)
                    );

                    EXIT.store(true, Ordering::Release);
                    break;
                }

                // Log progress every million attempts
                if count % 1_000_000 == 0 {
                    let time_secs = timer.elapsed().as_secs_f64();
                    let rate = if time_secs > 0.0 { count as f64 / time_secs } else { 0.0 };
                    logfather::info!(
                        "cpu {} progress: {} attempts in {:.1}s ({:.0} attempts/sec)",
                        i,
                        count.to_formatted_string(&Locale::en),
                        time_secs,
                        rate
                    );
                }
            }
        });
    }
}

fn get_validated_prefix_keypair(args: &KeypairArgs) -> &'static str {
    const BS58_CHARS: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    if let Some(ref prefix) = args.prefix {
        for c in prefix.chars() {
            assert!(
                BS58_CHARS.contains(c),
                "your prefix contains invalid bs58: {}",
                c
            );
        }
        let prefix = maybe_bs58_aware_lowercase(&prefix, args.case_insensitive);
        return prefix.leak();
    }
    ""
}

fn get_validated_suffix_keypair(args: &KeypairArgs) -> &'static str {
    const BS58_CHARS: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    if let Some(ref suffix) = args.suffix {
        for c in suffix.chars() {
            assert!(
                BS58_CHARS.contains(c),
                "your suffix contains invalid bs58: {}",
                c
            );
        }
        let suffix = maybe_bs58_aware_lowercase(&suffix, args.case_insensitive);
        return suffix.leak();
    }
    ""
}
