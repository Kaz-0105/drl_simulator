# 仮想環境の作り方

ルートディレクトリにて下記のコマンドを入力．

```
python -m venv venv 
source venv/Scripts/activate
```

これによりvenvフォルダが作成され，仮想環境の作成が可能になる．

次に下記コマンドを打ち込むことで，任意のモジュールを仮想環境上にインストールできる．

```
pip install {モジュール名}
```

終了するときは下記コマンドを入力．

```
deactivate
```

必要なパッケージは`requirements.txt`にまとめてある．
これらをまとめてインストールする場合は下記のコマンドを入力．

```
pip install -r requirements.txt
```

# yamlファイルの作り方
以下の記事を参照するとよい．

https://qiita.com/mnoguchi/items/75fb224918f452a409cd


# Pythonのバージョンについて
3.7以上にしてほしい．




