document.addEventListener('DOMContentLoaded', function() {
    const generateBtn = document.getElementById('generateBtn');
    const playerNamesInput = document.getElementById('playerNames');
    const roundsInput = document.getElementById('rounds');
    const allowFiveInput = document.getElementById('allowFive');
    const resultsSection = document.getElementById('results');
    const resultsContent = document.getElementById('resultsContent');
    const errorDiv = document.getElementById('error');
    
    generateBtn.addEventListener('click', generateTableGroups);
    
    async function generateTableGroups() {
        // エラーメッセージをクリア
        errorDiv.style.display = 'none';
        errorDiv.textContent = '';

        // プレイヤー名を取得
        const playerNames = playerNamesInput.value
            .split('\n')
            .map(name => name.trim())
            .filter(name => name.length > 0);

        const rounds = parseInt(roundsInput.value);
        const allowFive = allowFiveInput.checked;

        // バリデーション
        if (playerNames.length < 4) {
            showError('参加者は4人以上必要です');
            return;
        }

        if (rounds < 1) {
            showError('回数は1以上必要です');
            return;
        }

        // ボタン無効化 + ローディング表示
        generateBtn.disabled = true;
        generateBtn.textContent = '生成中...';
        resultsSection.style.display = 'none';
        resultsContent.innerHTML = '';

        // ローディングインジケータを表示
        const loadingDiv = document.getElementById('loading');
        loadingDiv.style.display = 'block';

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    players: playerNames,
                    rounds: rounds,
                    allow_five: allowFive
                })
            });

            const data = await response.json();

            if (response.ok) {
                displayResults(data.results);
            } else {
                showError(data.error || 'エラーが発生しました');
            }
        } catch (error) {
            showError('通信エラーが発生しました');
            console.error(error);
        } finally {
            // ボタン復帰
            generateBtn.disabled = false;
            generateBtn.textContent = '卓組を生成';
            loadingDiv.style.display = 'none';
        }
    }
    
    function showError(message) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        resultsSection.style.display = 'none';
    }
    
    function displayResults(results) {
        let html = `
            <div class="summary">
                <p>参加者: ${results.players}人、${results.rounds}節</p>
                <p>5人打ち: ${results.allow_five ? 'あり' : 'なし'}</p>
            </div>
        `;
        
        // 各節の卓数を確認
        let maxTables = 0;
        results.rounds_data.forEach(round => {
            if (round.tables.length > maxTables) {
                maxTables = round.tables.length;
            }
        });
        
        // 各節の各卓の情報を構築
        const tableInfo = {};
        for (let i = 1; i <= maxTables; i++) {
            tableInfo[`卓${i}`] = {};
        }
        tableInfo['待機'] = {};
        
        results.rounds_data.forEach(round => {
            // 各卓の情報を記録
            round.tables.forEach(table => {
                tableInfo[`卓${table.table_num}`][round.round_num] = table.players.join('<br>');
            });
            // 待機者の情報を記録
            if (round.waiting.length > 0) {
                tableInfo['待機'][round.round_num] = round.waiting.join('<br>');
            }
        });
        
        // テーブル形式で表示
        html += `<div class="results-table-container">`;
        html += `<table class="results-table">`;
        html += `<thead>`;
        html += `<tr>`;
        html += `<th class="table-name-header">卓</th>`;
        for (let i = 1; i <= results.rounds; i++) {
            html += `<th class="round-header">第${i}節</th>`;
        }
        html += `</tr>`;
        html += `</thead>`;
        html += `<tbody>`;
        
        // 各卓の行を作成
        for (let i = 1; i <= maxTables; i++) {
            html += `<tr>`;
            html += `<td class="table-name">卓${i}</td>`;
            for (let j = 1; j <= results.rounds; j++) {
                const players = tableInfo[`卓${i}`][j] || '';
                html += `<td class="players-cell">${players}</td>`;
            }
            html += `</tr>`;
        }
        
        // 待機者の行を作成（待機者がいる場合のみ）
        let hasWaiting = false;
        for (let i = 1; i <= results.rounds; i++) {
            if (tableInfo['待機'][i]) {
                hasWaiting = true;
                break;
            }
        }
        
        if (hasWaiting) {
            html += `<tr>`;
            html += `<td class="table-name waiting-row">待機</td>`;
            for (let i = 1; i <= results.rounds; i++) {
                const waitingPlayers = tableInfo['待機'][i] || '';
                html += `<td class="waiting-cell">${waitingPlayers}</td>`;
            }
            html += `</tr>`;
        }
        
        html += `</tbody>`;
        html += `</table>`;
        html += `</div>`;
        
        // 統計情報
        html += `<div class="statistics">`;
        html += `<h3>統計情報</h3>`;
        
        // ペア統計
        const pairStats = results.statistics.pair_statistics;
        console.log('Pair Statistics:', pairStats);  // デバッグ用
        
        html += `<div class="pair-stats">`;
        html += `<p>最小同卓回数: ${pairStats.min_count}回</p>`;
        html += `<p>最大同卓回数: ${pairStats.max_count}回</p>`;
        html += `<p>カバレッジ: ${pairStats.coverage.toFixed(1)}%</p>`;
        
        // 同卓回数の分布
        const sortedCounts = Object.keys(pairStats.distribution).sort((a, b) => a - b);
        sortedCounts.forEach(count => {
            html += `<div class="stat-item">${count}回同卓: ${pairStats.distribution[count]}ペア</div>`;
        });
        
        // 0回同卓の警告
        if (pairStats.min_count === 0) {
            html += `<div class="warning" style="color: red; margin-top: 10px;">`;
            html += `⚠️ 注意: 一度も同卓しないペアが存在します`;
            html += `</div>`;
        }
        
        html += `</div>`;
        
        // 待機回数統計（5人で5人打ちなしの場合）
        if (results.statistics.wait_statistics) {
            html += `<div class="wait-stats">`;
            html += `<h4>待機回数統計</h4>`;
            results.statistics.wait_statistics.forEach(stat => {
                html += `<div class="wait-stat-item">${stat.player}: ${stat.count}回</div>`;
            });
            html += `</div>`;
        }
        
        html += `</div>`;
        
        resultsContent.innerHTML = html;
        resultsSection.style.display = 'block';
    }
});