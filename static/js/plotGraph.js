document.addEventListener('DOMContentLoaded', function () {
    // Fetch the latest data from the high_ev_bets endpoint
    fetch('/data/high_ev_bets')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error("Error fetching high_ev_bets data:", data.error);
                return;
            }
            
            // Extract the implied probabilities and expected values from the fetched data
            const impliedProbs = data.map(entry => entry.implied_prob_home || entry.implied_prob_away);
            const expectedValues = data.map(entry => entry.expected_value);

            // Fetch the latest optimal bet data to highlight the optimal bet
            fetch('/data/optimal_bet')
                .then(response => response.json())
                .then(optimalData => {
                    if (optimalData.error) {
                        console.error("Error fetching optimal_bet data:", optimalData.error);
                        return;
                    }
                    
                    // Locate the same match in the high_ev_bets data
                    const optimalMatch = optimalData['Closest Match'];
                    const optimalBet = data.find(entry => 
                        entry.home_team === optimalMatch.home_team &&
                        entry.away_team === optimalMatch.away_team &&
                        entry.commence_time === optimalMatch.commence_time
                    );

                    if (!optimalBet) {
                        console.error("Optimal bet match not found in high_ev_bets data.");
                        return;
                    }

                    const optimalPoint = {
                        x: optimalBet.implied_prob_home || optimalBet.implied_prob_away,
                        y: optimalBet.expected_value
                    };

                    // Create the scatter plot graph using Chart.js
                    const ctx = document.getElementById('evImpliedGraph').getContext('2d');
                    const evImpliedChart = new Chart(ctx, {
                        type: 'scatter',
                        data: {
                            datasets: [{
                                label: 'Expected Value vs Implied Probability',
                                data: impliedProbs.map((prob, index) => ({
                                    x: prob,
                                    y: expectedValues[index]
                                })).concat(optimalPoint),
                                backgroundColor: impliedProbs.map((_, index) =>
                                    impliedProbs[index] === optimalPoint.x && expectedValues[index] === optimalPoint.y ? 'red' : 'rgba(0, 123, 255, 0.7)'
                                ).concat('red'),
                                pointRadius: impliedProbs.map((_, index) =>
                                    impliedProbs[index] === optimalPoint.x && expectedValues[index] === optimalPoint.y ? 8 : 5
                                ).concat(8),
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: {
                                    type: 'linear',
                                    position: 'bottom',
                                    title: {
                                        display: true,
                                        text: 'Implied Probability'
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Expected Value'
                                    }
                                }
                            },
                            plugins: {
                                legend: {
                                    display: false
                                }
                            }
                        }
                    });
                })
                .catch(error => {
                    console.error("Error fetching optimal_bet data:", error);
                });
        })
        .catch(error => {
            console.error("Error fetching high_ev_bets_data:", error);
        });
});
