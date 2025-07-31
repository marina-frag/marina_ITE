## Running

for windows run:
```bash
cmd.exe /c run_experiments.bat
```

for linux run:
```bash
chmod +x run_experiments.sh and ./run_experiments.sh
 ```
The any current work in bash is then saved in results. After running the above we have our first results.  We rename said results. results->statistics. Then we run:

then :
```bash
python test_stats.py
```
Again the results are saved in results. Then passed to notebook to compare with other hyperparemeter runs

## Using aws:

Firstly connect with ssh:
```bash
ssh -i "C:\Users\marin\Documents\ITE\PROJECT\SettingUp\marina_aws_key_pair.pem" ubuntu@ec2-13-60-22-71.eu-north-1.compute.amazonaws.com
```
Clone repository:
```bash
git clone https://github.com/marina-frag/marina_ITE.git
```
Create session:
```bash
tmux new -s session_name
```
To disconnect:
```bash
Cntrl+b d
```
To reconnect:
```bash
tmux attach -t session_name 
```
To terminate session:
```bash
tmux kill-session -t session_name 
```
See existing sessions:
```bash
tmux ls
```
