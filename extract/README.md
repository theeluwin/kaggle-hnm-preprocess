# H&M 벡터 추출기

## 데이터 준비

`data/` 폴더 아래에 `df_item_preprocessed.pq` 넣어주기

## 도커 빌드

```bash
./scripts/build.sh
```

## 실행 스크립트 준비

`x.sh`:

```bash
runpy () {
    docker run \
        -it \
        --rm \
        --init \
        --gpus '"device=0"' \
        --shm-size 16G \
        --volume="$HOME/.cache/torch:/root/.cache/torch" \
        --volume="$PWD:/workspace" \
        --volume="$PWD/../raw:/workspace/rough" \
        ids-rec-hnm-extract \
        python "$@"
}

runpy extract_image_vector.py
runpy extract_text_vector.py
```

## 실행 순서

1. `preresize.py`
2. `extract_image_vectors.py`
3. `extract_text_vectors.py`
4. `concat_with_noagg.py`
