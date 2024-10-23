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
                    
                    // Find the index of the optimal bet
                    const optimalIndex = impliedProbs.findIndex(prob => 
                        prob === optimalData['Closest Match'].implied_probability
                    );

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
                                })),
                                backgroundColor: impliedProbs.map((_, index) =>
                                    index === optimalIndex ? 'red' : 'rgba(0, 123, 255, 0.7)'
                                ),
                                pointRadius: impliedProbs.map((_, index) =>
                                    index === optimalIndex ? 8 : 5
                                ),
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
