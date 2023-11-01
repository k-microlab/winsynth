use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Condvar, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;
use num::{Bounded, FromPrimitive, Num, ToPrimitive};
use winapi::shared::minwindef::{DWORD, UINT};
use winapi::shared::mmreg::{WAVE_FORMAT_PCM, WAVEFORMATEX};
use winapi::shared::winerror::S_OK;
use winapi::um::mmeapi::{waveOutClose, waveOutGetDevCapsW, waveOutGetNumDevs, waveOutOpen, waveOutPrepareHeader, waveOutUnprepareHeader, waveOutWrite};
use winapi::um::mmsystem::{CALLBACK_FUNCTION, HWAVEOUT, MM_WOM_CLOSE, MM_WOM_DONE, MM_WOM_OPEN, WAVEHDR, WAVEOUTCAPSW};

pub const DEFAULT_SAMPLE_RATE: u32 = 44100;
pub const DEFAULT_CHANNELS: usize = 2;
pub const PI: f64 = std::f64::consts::PI;
pub const TAU: f64 = std::f64::consts::TAU;

fn main() {
    let source = WaveSource::<i16, 8, 512>::new(|time, chanel| {
        if chanel == 1 {
            (50.0 * TAU * time).sin().signum()
        } else {
            (50.0 * TAU * time).sin()
        }
    });
    let _ = source.start(DEFAULT_SAMPLE_RATE, DEFAULT_CHANNELS);

    loop {
        std::thread::sleep(Duration::from_secs(1000));
    }
}

struct WaveSource<T, const BLOCKS: usize, const SAMPLES: usize> where T: Num + Bounded + FromPrimitive + ToPrimitive + Copy + Send + Sync + 'static {
    source: Box<dyn Fn(f64, usize) -> f64 + Send + Sync + 'static>,
    ready: Arc<AtomicBool>,
    free: Arc<Mutex<usize>>,
    non_zero: Arc<Condvar>,
    block_memory: Box<[[[T; DEFAULT_CHANNELS]; SAMPLES]; BLOCKS]>,
    wave_headers: Box<[WAVEHDR; BLOCKS]>
}

unsafe impl<T, const BLOCKS: usize, const SAMPLES: usize> Send for WaveSource<T, BLOCKS, SAMPLES> where T: Num + Bounded + FromPrimitive + ToPrimitive + Copy + Send + Sync + 'static {}
unsafe impl<T, const BLOCKS: usize, const SAMPLES: usize> Sync for WaveSource<T, BLOCKS, SAMPLES> where T: Num + Bounded + FromPrimitive + ToPrimitive + Copy + Send + Sync + 'static {}

struct ThreadedCallback {
    free: Arc<Mutex<usize>>,
    non_zero: Arc<Condvar>
}

impl DeviceCallback for ThreadedCallback {
    fn done(&mut self) {
        let mut free = self.free.lock().unwrap();
        *free += 1;
        self.non_zero.notify_one();
    }
}

impl<T, const BLOCKS: usize, const SAMPLES: usize> WaveSource<T, BLOCKS, SAMPLES> where T: Num + Bounded + FromPrimitive + ToPrimitive + Copy + Send + Sync {
    pub fn new<S>(source: S) -> Self where S: 'static + Send + Sync + Fn(f64, usize) -> f64 {
        let mut this = Self {
            source: Box::new(source),
            ready: Arc::new(AtomicBool::new(true)),
            free: Arc::new(Mutex::new(BLOCKS)),
            non_zero: Arc::new(Default::default()),
            block_memory: Box::new([[[T::zero(); DEFAULT_CHANNELS]; SAMPLES]; BLOCKS]),
            wave_headers: Box::new([WAVEHDR::default(); BLOCKS])
        };

        for (n, header) in this.wave_headers.iter_mut().enumerate() {
            header.dwBufferLength = (SAMPLES * std::mem::size_of::<T>() * DEFAULT_CHANNELS) as _;
            header.lpData = this.block_memory[n].as_mut_ptr() as _;
        }

        this
    }

    pub fn start(mut self, sample_rate: u32, channels: usize) -> JoinHandle<()> {
        let devices = enumerate_devices();
        let device = devices.into_iter().next().expect("No output devices found");

        let non_zero = Arc::clone(&self.non_zero);
        let non_zero_s = Arc::clone(&self.non_zero);
        let free = Arc::clone(&self.free);

        let mut open_device = device.open::<T>(sample_rate, channels, Box::new(Box::new(ThreadedCallback {
            free, non_zero
        })));

        let handle = std::thread::spawn(move || {
            let mut global_time = 0.0;
            let time_step = 1.0 / sample_rate as f64;
            let max_sample = T::max_value().to_f64().expect(&format!("Failed to convert {} tp f64", std::any::type_name::<T>()));
            // let mut previous_sample = T::zero();
            let mut current_block = 0;

            while self.ready.load(Ordering::SeqCst) {
                let mut free = self.free.lock().unwrap();
                *if *free == 0 {
                    self.non_zero.wait(free).unwrap()
                } else {
                    free
                } -= 1;


                let current_header = &mut self.wave_headers[current_block];

                const WHDR_PREPARED: DWORD = 2;

                if (current_header.dwFlags & WHDR_PREPARED) > 0 {
                    unsafe {
                        waveOutUnprepareHeader(open_device.device.inner, current_header, std::mem::size_of::<WAVEHDR>() as _);
                    }
                }

                for n in 0..SAMPLES {
                    for c in 0..DEFAULT_CHANNELS {
                        let sample = (self.source)(global_time, c).clamp(-1.0, 1.0) * max_sample;
                        let sample = T::from_f64(sample).expect(&format!("Cannot convert sample value {} to {}", sample, std::any::type_name::<T>()));
                        self.block_memory[current_block][n][c] = sample;
                        // previous_sample = sample;
                    }
                    global_time += time_step;
                }

                unsafe {
                    waveOutPrepareHeader(open_device.device.inner, current_header, std::mem::size_of::<WAVEHDR>() as _);
                    waveOutWrite(open_device.device.inner, current_header, std::mem::size_of::<WAVEHDR>() as _);
                }
                current_block += 1;
                current_block %= BLOCKS;
            }
        });

        non_zero_s.notify_one();

        handle
    }

    fn main_thread(&mut self) {

    }
}

fn enumerate_devices() -> Vec<Device> {
    let device_count = unsafe { waveOutGetNumDevs() };
    let mut woc = WAVEOUTCAPSW::default();
    let mut devices = Vec::with_capacity(device_count as usize);
    for id in 0..device_count {
        if unsafe { waveOutGetDevCapsW(id as _, &mut woc, std::mem::size_of::<WAVEOUTCAPSW>() as _) } == S_OK as _ {
            devices.push(Device {
                id,
                caps: woc.clone(),
                inner: std::ptr::null_mut()
            });
        }
    }
    devices
}

pub trait DeviceCallback {
    fn open(&mut self) {}
    fn close(&mut self) {}
    fn done(&mut self);
}

pub struct Device {
    id: UINT,
    caps: WAVEOUTCAPSW,
    inner: HWAVEOUT
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    pub fn get_name(&self) -> String {
        let name = self.caps.szPname;
        let end = name.iter().position(|&c| c == 0).unwrap_or(32);
        String::from_utf16(&name[0..end]).expect("Invalid device name")
    }

    pub fn open<T>(mut self, sample_rate: u32, channels: usize, callback: Box<Box<dyn DeviceCallback>>) -> OpenDevice<T> where T: Num + Bounded  {
        let mut format = WAVEFORMATEX::default();
        let bytes = std::mem::size_of::<T>();
        format.wFormatTag = WAVE_FORMAT_PCM;
        format.nSamplesPerSec = sample_rate as _;
        format.wBitsPerSample = bytes as u16 * 8;
        format.nChannels = channels as _;
        format.nBlockAlign = (bytes * channels) as _;
        format.nAvgBytesPerSec = sample_rate * format.nBlockAlign as u32;
        format.cbSize = 0;

        let mut device = Box::new(self);

        if unsafe { waveOutOpen(&mut device.inner, device.id, &format, Self::callback as _, Box::leak(callback) as *const _ as _, CALLBACK_FUNCTION) } != S_OK as _ {
            panic!("Device opening failed");
        }

        let mut open = OpenDevice {
            device,
            ty: PhantomData
        };

        open
    }

    extern "system" fn callback(_: HWAVEOUT, msg: UINT, mut callback: ManuallyDrop<Box<Box<dyn DeviceCallback>>>, p1: DWORD, p2: DWORD) {
        match msg {
            MM_WOM_OPEN => callback.open(),
            MM_WOM_CLOSE => callback.close(),
            MM_WOM_DONE => callback.done(),
            other => {
                println!("Unknown message: {}", other)
            }
        }
    }
}

pub struct OpenDevice<T> where T: Num + Bounded  {
    device: Box<Device>,
    ty: PhantomData<T>
}

impl<T> OpenDevice<T> where T: Num + Bounded {

}

impl<T> Drop for OpenDevice<T> where T: Num + Bounded {
    fn drop(&mut self) {
        unsafe {
            if waveOutClose(self.device.inner) != S_OK as _ {
                eprintln!("Device close failed")
            }
        }
    }
}